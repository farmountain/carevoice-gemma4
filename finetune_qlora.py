# %% [markdown]
# # CareVoice — QLoRA Fine-tuning on SurgWound + SPRSound
#
# Fine-tunes Gemma 4 4B-IT with QLoRA on:
# - **SurgWound** (697 surgical wound images, green/yellow/red urgency GT)
# - **SPRSound** (2,683 pediatric respiratory recordings, Normal/Abnormal/CAS/DAS GT)
#
# Output: LoRA adapter saved to `/kaggle/working/carevoice-lora/`
# Load in inference notebook: `PeftModel.from_pretrained(base_model, adapter_path)`

# %% [markdown]
# ## 0 — Install dependencies

# %%
# !pip install -q peft==0.14.0 trl==0.16.0 bitsandbytes accelerate datasets

# %% [markdown]
# ## 1 — Imports & constants

# %%
import os, json, random, base64, io, math, re
import numpy as np
import torch
from pathlib import Path

# ── Model & adapter paths ─────────────────────────────────────────────────────
GEMMA_MODEL_PATH = "/kaggle/input/gemma-4/transformers/gemma-4-e4b-it/1"
ADAPTER_OUT      = "/kaggle/working/carevoice-lora"
RESULTS_OUT      = "/kaggle/working/finetune_results.json"

# ── Training hyper-parameters ─────────────────────────────────────────────────
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TRAIN_EPOCHS    = 3
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 1
GRAD_ACCUM      = 4          # effective batch = 4
MAX_SEQ_LEN     = 1024
SEED            = 42

# ── Dataset limits (balanced per class) ──────────────────────────────────────
MAX_IMG_TRAIN   = 450        # ~150 per urgency level
MAX_IMG_VAL     = 90         # ~30 per urgency level
MAX_AUD_TRAIN   = 200        # ~100 Normal + 100 Abnormal
MAX_AUD_VAL     = 40

# ── SPRSound dataset path (auto-clone if not present) ────────────────────────
import subprocess as _sp
_SPRSOUND_CANDIDATES = [
    Path("/kaggle/input/sprsound"),
    Path("/workspace/datasets/sprsound"),
    Path("/tmp/sprsound"),
]
SPRSOUND_DIR = next((p for p in _SPRSOUND_CANDIDATES if p.exists()), None)
if SPRSOUND_DIR is None:
    SPRSOUND_DIR = Path("/kaggle/working/sprsound")
    print("Cloning SPRSound (CC BY 4.0, ~4.4 GB) — 3-5 min...")
    _sp.run(
        f"git clone --depth 1 https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound {SPRSOUND_DIR}",
        shell=True, check=True,
    )
print(f"SPRSound dir: {SPRSOUND_DIR}")
AUDIO_SR = 16000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

# %% [markdown]
# ## 2 — Load model in 4-bit (QLoRA)

# %%
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Gemma4ForConditionalGeneration

print("Loading processor...")
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_PATH)

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = Gemma4ForConditionalGeneration.from_pretrained(
    GEMMA_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# %% [markdown]
# ## 3 — Apply LoRA

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

def find_target_modules(model):
    """Find all linear layer name suffixes in the LM backbone to apply LoRA to."""
    target_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"}
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in target_suffixes:
                found.add(suffix)
    return sorted(found)

target_modules = find_target_modules(model)
print("LoRA target modules:", target_modules)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=target_modules,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 4 — System prompts (identical to inference notebook)

# %%
IMAGE_SYSTEM = """You are CareVoice, triaging a patient-submitted clinical photo.
green = routine care  |  yellow = evaluate within hours  |  red = emergency now

IMPORTANT: Your ENTIRE response must be valid JSON. Start with { and end with }.

{
  "response": "<findings in plain language>",
  "visual_findings": {
    "image_type": "wound",
    "description": "<describe: wound size, colour, edges, signs of infection>",
    "severity_indicators": [],
    "differential": []
  },
  "extracted_info": {
    "chief_complaint": "<one-sentence summary>",
    "symptoms": [],
    "urgent": false,
    "escalation_reason": null,
    "triage_level": "green"
  },
  "intake_complete": false,
  "follow_up_questions": []
}

Triage rules:
- green: clean intact sutured wound; minor abrasion; no edge separation; no infection signs
- yellow: erythema, swelling, crusting, unclear depth; evaluate within hours
- red: dehiscence (separated/gaping edges); active bleeding; exposed bone/tendon; severe burns; systemic infection"""

AUDIO_SYSTEM = """You are CareVoice listening to patient audio.
If respiratory: describe clinically as JSON only.

{
  "response": "<acknowledgement>",
  "audio_analysis": {
    "audio_type": "speech|cough|breathing|wheeze|other",
    "transcription": null,
    "clinical_observations": "",
    "respiratory_findings": {
      "cough_present": false,
      "wheeze_present": false,
      "stridor_present": false,
      "abnormal_breathing": false
    }
  },
  "extracted_info": {
    "chief_complaint": null,
    "symptoms": [],
    "urgent": false,
    "escalation_reason": null,
    "triage_level": "green"
  },
  "intake_complete": false
}"""

# %% [markdown]
# ## 5 — SurgWound dataset (image triage)

# %%
from datasets import load_dataset
from PIL import Image

def parse_gt(answer: str) -> str:
    a = answer.lower()
    if "red" in a or "emergency" in a or "immediate" in a:
        return "red"
    if "yellow" in a or "evaluate" in a or "hours" in a:
        return "yellow"
    return "green"

def decode_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

def make_image_response(gt: str) -> str:
    urgent = gt == "red"
    return json.dumps({
        "response": f"Wound assessed as {gt} priority.",
        "visual_findings": {
            "image_type": "wound",
            "description": f"Wound consistent with {gt} urgency.",
            "severity_indicators": ["dehiscence", "open edges"] if gt == "red" else
                                   ["erythema", "swelling"]    if gt == "yellow" else
                                   ["intact margins"],
            "differential": ["wound dehiscence"]      if gt == "red"    else
                            ["surgical site infection"] if gt == "yellow" else
                            ["healing wound"],
        },
        "extracted_info": {
            "chief_complaint": "wound triage",
            "symptoms": [],
            "urgent": urgent,
            "escalation_reason": "Wound requires immediate attention." if urgent else None,
            "triage_level": gt,
        },
        "intake_complete": False,
        "follow_up_questions": ["Has the wound appearance changed recently?"],
    }, ensure_ascii=False)

print("Loading SurgWound...")
surgwound = load_dataset("xuxuxuxuxu/SurgWound", split="train")
print(f"  Total samples: {len(surgwound)}")

sw_by_level = {"green": [], "yellow": [], "red": []}
for item in surgwound:
    sw_by_level[parse_gt(item["answer"])].append(item)
for lvl, items in sw_by_level.items():
    print(f"  {lvl}: {len(items)}")

n_per_level_train = min(len(sw_by_level[l]) for l in sw_by_level)
n_per_level_train = min(n_per_level_train, MAX_IMG_TRAIN // 3)
n_per_level_val   = min(
    min(len(sw_by_level[l]) for l in sw_by_level) - n_per_level_train,
    MAX_IMG_VAL // 3
)

img_train, img_val = [], []
for lvl in ["green", "yellow", "red"]:
    items = sw_by_level[lvl].copy()
    random.shuffle(items)
    img_train.extend(items[:n_per_level_train])
    img_val.extend(items[n_per_level_train : n_per_level_train + n_per_level_val])

random.shuffle(img_train)
random.shuffle(img_val)
print(f"\nImage — train: {len(img_train)}  val: {len(img_val)}")

# %% [markdown]
# ## 6 — SPRSound dataset (audio)

# %%
import scipy.io.wavfile as wf

_ABNORMAL_LABELS = {"Abnormal", "CAS", "DAS", "Wheeze", "Crackle", "Rhonchi", "Stridor",
                    "wheeze", "crackle", "rhonchi", "stridor"}

def load_wav_array(path: str) -> np.ndarray:
    try:
        sr, data = wf.read(path)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data = data / float(np.iinfo(np.int16).max)
        return data
    except Exception:
        return np.zeros(AUDIO_SR, dtype=np.float32)

def make_audio_response(gt_label: str) -> str:
    is_abnormal = gt_label in _ABNORMAL_LABELS
    is_wheeze   = gt_label in {"CAS", "Wheeze", "wheeze", "Rhonchi", "rhonchi"}
    is_stridor  = gt_label in {"Stridor", "stridor"}
    is_crackle  = gt_label in {"DAS", "Crackle", "crackle"}
    return json.dumps({
        "response": "Abnormal respiratory sound detected." if is_abnormal else "Normal breath sounds.",
        "audio_analysis": {
            "audio_type": "wheeze" if is_wheeze else "breathing",
            "transcription": None,
            "clinical_observations": (
                "Continuous adventitious sounds (wheeze/rhonchus) detected." if is_wheeze else
                "Discontinuous adventitious sounds (crackle) detected."       if is_crackle else
                "Stridor — upper airway obstruction concern."                  if is_stridor else
                "Normal breath sounds. No adventitious sounds detected."
            ),
            "respiratory_findings": {
                "cough_present":     not is_abnormal,
                "wheeze_present":    is_wheeze,
                "stridor_present":   is_stridor,
                "abnormal_breathing": is_abnormal,
            },
        },
        "extracted_info": {
            "chief_complaint": "wheeze/bronchospasm" if is_wheeze else
                               "stridor/airway obstruction" if is_stridor else
                               "respiratory assessment",
            "symptoms": ["wheeze"] if is_wheeze else ["stridor"] if is_stridor else [],
            "urgent": is_stridor,
            "escalation_reason": "Stridor detected." if is_stridor else None,
            "triage_level": "red" if is_stridor else "yellow" if is_abnormal else "green",
        },
        "intake_complete": False,
    }, ensure_ascii=False)

print("Scanning SPRSound annotations...")
ann_files = sorted(SPRSOUND_DIR.glob("**/*.json"))
print(f"  Annotation files: {len(ann_files)}")

_all_labeled = []
for af in ann_files:
    try:
        ann = json.loads(af.read_text(encoding="utf-8"))
        gt_label = ann.get("record_annotation", "")
        wavs = list(SPRSOUND_DIR.glob(f"**/{af.stem}.wav"))
        if wavs and gt_label and "Poor" not in gt_label:
            _all_labeled.append({"wav": str(wavs[0]), "gt_label": gt_label})
    except Exception:
        pass

aud_by_bucket: dict = {"Normal": [], "Abnormal": []}
for s in _all_labeled:
    bucket = "Abnormal" if s["gt_label"] in _ABNORMAL_LABELS else "Normal"
    aud_by_bucket[bucket].append(s)

for k, v in aud_by_bucket.items():
    print(f"  {k}: {len(v)}")

n_aud_per_class_train = min(
    MAX_AUD_TRAIN // 2,
    min(len(aud_by_bucket["Normal"]), len(aud_by_bucket["Abnormal"]))
)
n_aud_per_class_val = max(5, n_aud_per_class_train // 5)

aud_train, aud_val = [], []
for bucket in ["Normal", "Abnormal"]:
    items = aud_by_bucket[bucket].copy()
    random.shuffle(items)
    aud_train.extend(items[:n_aud_per_class_train])
    aud_val.extend(items[n_aud_per_class_train : n_aud_per_class_train + n_aud_per_class_val])

random.shuffle(aud_train)
random.shuffle(aud_val)
print(f"\nAudio — train: {len(aud_train)}  val: {len(aud_val)}")

# %% [markdown]
# ## 7 — PyTorch Dataset

# %%
from torch.utils.data import Dataset as TorchDataset

class CareVoiceDataset(TorchDataset):
    """Mixed image + audio SFT dataset for CareVoice QLoRA fine-tuning."""

    def __init__(self, img_samples: list, aud_samples: list, processor, max_len: int = MAX_SEQ_LEN):
        self.processor = processor
        self.max_len   = max_len
        self.items     = (
            [{"modality": "image", **s} for s in img_samples] +
            [{"modality": "audio", **s} for s in aud_samples]
        )
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            return self._image_item(item) if item["modality"] == "image" else self._audio_item(item)
        except Exception as e:
            print(f"  Sample {idx} error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def _apply_labels_mask(self, inputs, assistant_text: str) -> dict:
        """Mask all tokens except the assistant response (train only on outputs)."""
        input_ids = inputs["input_ids"].squeeze(0)
        labels    = input_ids.clone()
        resp_len  = len(self.processor.tokenizer.encode(assistant_text, add_special_tokens=False))
        non_pad   = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
        mask_end  = max(0, non_pad - resp_len)
        labels[:mask_end] = -100   # mask everything before assistant response
        labels[non_pad:]  = -100   # mask padding
        result = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        result["labels"] = labels
        return result

    def _image_item(self, item: dict) -> dict:
        gt   = parse_gt(item["answer"])
        img  = decode_pil(item["image"])
        asst = make_image_response(gt)
        msgs = [
            {"role": "system",    "content": IMAGE_SYSTEM},
            {"role": "user",      "content": [{"type": "image"},
                                               {"type": "text", "text": "Triage this wound."}]},
            {"role": "assistant", "content": asst},
        ]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt",
                                truncation=True, max_length=self.max_len, padding="max_length")
        return self._apply_labels_mask(inputs, asst)

    def _audio_item(self, item: dict) -> dict:
        arr  = load_wav_array(item["wav"])
        asst = make_audio_response(item["gt_label"])
        msgs = [
            {"role": "system",    "content": AUDIO_SYSTEM},
            {"role": "user",      "content": [{"type": "audio"},
                                               {"type": "text", "text": "Analyze this respiratory sound."}]},
            {"role": "assistant", "content": asst},
        ]
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=prompt, audios=[(arr, AUDIO_SR)], return_tensors="pt",
                                truncation=True, max_length=self.max_len, padding="max_length")
        return self._apply_labels_mask(inputs, asst)


train_ds = CareVoiceDataset(img_train, aud_train, processor)
val_ds   = CareVoiceDataset(img_val,   aud_val,   processor)
print(f"Dataset — train: {len(train_ds)}  val: {len(val_ds)}")

# %% [markdown]
# ## 8 — Train

# %%
from transformers import TrainingArguments, Trainer


def data_collator(features: list) -> dict:
    """Stack tensors; leave variable-size (pixel_values, audio) as lists."""
    batch = {}
    for key in features[0]:
        vals = [f[key] for f in features]
        if isinstance(vals[0], torch.Tensor):
            try:
                batch[key] = torch.stack(vals)
            except RuntimeError:
                batch[key] = vals   # variable-size tensors: leave as list
        else:
            batch[key] = vals
    return batch


training_args = TrainingArguments(
    output_dir=ADAPTER_OUT,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    label_names=["labels"],
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
print(f"Training — {len(train_ds)} samples | {TRAIN_EPOCHS} epochs | "
      f"{steps_per_epoch * TRAIN_EPOCHS} total steps")

train_result = trainer.train()
print(f"\nTrain loss: {train_result.training_loss:.4f}  "
      f"Runtime: {train_result.metrics.get('train_runtime', 0):.0f}s")

# %% [markdown]
# ## 9 — Save LoRA adapter

# %%
print(f"Saving adapter to {ADAPTER_OUT} ...")
model.save_pretrained(ADAPTER_OUT)
processor.save_pretrained(ADAPTER_OUT)

results = {
    "train_loss":       train_result.training_loss,
    "train_runtime_s":  train_result.metrics.get("train_runtime", 0),
    "img_train":        len(img_train),
    "aud_train":        len(aud_train),
    "epochs":           TRAIN_EPOCHS,
    "lora_r":           LORA_R,
    "lora_alpha":       LORA_ALPHA,
    "target_modules":   target_modules,
}
Path(RESULTS_OUT).write_text(json.dumps(results, indent=2))

print("Adapter files:")
for f in sorted(Path(ADAPTER_OUT).glob("*")):
    print(f"  {f.name}: {f.stat().st_size // 1024} KB")

# %% [markdown]
# ## 10 — Quick validation (10 images + 6 audio)

# %%
from peft import PeftModel

print("Loading adapter for quick validation...")
model_eval = Gemma4ForConditionalGeneration.from_pretrained(
    GEMMA_MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", torch_dtype=torch.bfloat16,
)
model_eval = PeftModel.from_pretrained(model_eval, ADAPTER_OUT)
model_eval.eval()

_PAT = re.compile(r"(\{.*\})", re.DOTALL)

def extract_triage(text: str) -> str:
    m = _PAT.search(text)
    if m:
        try:
            return json.loads(m.group(1)).get("extracted_info", {}).get("triage_level", "none") or "none"
        except Exception:
            pass
    for lbl in ["red", "yellow", "green"]:
        if f'"{lbl}"' in text:
            return lbl
    return "none"

def extract_abnormal(text: str) -> bool:
    m = _PAT.search(text)
    if m:
        try:
            rf = json.loads(m.group(1)).get("audio_analysis", {}).get("respiratory_findings", {})
            return rf.get("abnormal_breathing", False) or rf.get("wheeze_present", False)
        except Exception:
            pass
    return False

# ── Image validation ──────────────────────────────────────────────────────────
img_correct = img_total = 0
for item in img_val[:10]:
    gt  = parse_gt(item["answer"])
    img = decode_pil(item["image"])
    msgs = [{"role": "system", "content": IMAGE_SYSTEM},
            {"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "Triage this wound."}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp    = processor(text=prompt, images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model_eval.generate(**inp, max_new_tokens=400, do_sample=False,
                                  pad_token_id=processor.tokenizer.eos_token_id)
    pred  = extract_triage(processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
    match = pred == gt
    img_correct += int(match)
    img_total   += 1
    print(f"  Image  GT={gt:6s}  pred={pred:6s}  {'✅' if match else '❌'}")

print(f"\nImage val (10 samples): {img_correct}/{img_total} = {img_correct / img_total * 100:.0f}%")

# ── Audio validation ──────────────────────────────────────────────────────────
aud_correct = aud_total = 0
for item in aud_val[:6]:
    gt_label  = item["gt_label"]
    gt_bucket = "Abnormal" if gt_label in _ABNORMAL_LABELS else "Normal"
    arr       = load_wav_array(item["wav"])
    msgs = [{"role": "system", "content": AUDIO_SYSTEM},
            {"role": "user", "content": [{"type": "audio"},
                                          {"type": "text", "text": "Analyze this respiratory sound."}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp    = processor(text=prompt, audios=[(arr, AUDIO_SR)], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model_eval.generate(**inp, max_new_tokens=300, do_sample=False,
                                  pad_token_id=processor.tokenizer.eos_token_id)
    pred_abn    = extract_abnormal(processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
    pred_bucket = "Abnormal" if pred_abn else "Normal"
    match = pred_bucket == gt_bucket
    aud_correct += int(match)
    aud_total   += 1
    print(f"  Audio  GT={gt_label:10s} ({gt_bucket:8s})  pred={pred_bucket:8s}  {'✅' if match else '❌'}")

print(f"\nAudio val (6 samples): {aud_correct}/{aud_total} = {aud_correct / aud_total * 100:.0f}%")

# ── Save final results ────────────────────────────────────────────────────────
results.update({
    "img_val_accuracy_10": img_correct / max(img_total, 1),
    "aud_val_accuracy_6":  aud_correct / max(aud_total, 1),
})
Path(RESULTS_OUT).write_text(json.dumps(results, indent=2))

print("\n✅ Fine-tuning complete.")
print(f"   Adapter:  {ADAPTER_OUT}")
print(f"   Results:  {RESULTS_OUT}")
print("   Load with: PeftModel.from_pretrained(base_model, '/kaggle/input/carevoice-lora')")
