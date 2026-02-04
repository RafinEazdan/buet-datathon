# 🎯 Bengali Speech Recognition - Multi-Part Training Guide

## Overview

This guide explains how to train the wav2vec2 model for Bengali speech recognition across **5 separate Kaggle sessions** to work around the 12-hour time limit.

---

## 📅 Training Timeline

| Part | Duration | Training Stage | Epochs | Output File |
|------|----------|----------------|--------|-------------|
| **1** | ~3 hours | Phase 1 (Start) | 1-18 | `checkpoint_part1.safetensors` |
| **2** | ~3 hours | Phase 1 (Continue) | 19-36 | `checkpoint_part2.safetensors` |
| **3** | ~3 hours | Phase 1 (Continue) | 37-54 | `checkpoint_part3.safetensors` |
| **4** | ~4 hours | Phase 1 (Final) + Phase 2 | 55-70 + 1-7 | `checkpoint_final.safetensors` |
| **5** | ~1 hour | Inference Only | N/A | `submission.csv` |

**Total Time**: ~14 hours (split across 5 sessions)

---

## 🚀 Detailed Instructions

### **Part 1: Initial Training**

#### Setup
1. Open `fine-tune.ipynb` in Kaggle
2. Enable GPU (P100 or T4)
3. Attach the Bengali speech dataset

#### Configuration
Find the configuration cell and set:
```python
TRAINING_PART = 1
```

#### Run Training
1. Click "Run All" or execute cells sequentially
2. Wait ~3 hours for training to complete
3. Monitor the output for progress

#### Download Files
After training completes, download:
- `/kaggle/working/checkpoint_part1.safetensors` (~1.2 GB)
- `/kaggle/working/processor_part1/` (folder with vocabulary files)
- `/kaggle/working/metrics_part1.png` (optional - training plots)

#### Save These Files
Store these files safely on your local machine. You'll need them for Part 2.

---

### **Part 2: Continue Training**

#### Start Fresh Session
1. **Important**: Start a NEW Kaggle session
2. The old `/kaggle/working/` directory should be empty
3. Enable GPU again

#### Upload Checkpoint
1. Click "Add Data" → "Upload"
2. Create a new dataset named `model-checkpoint`
3. Upload `checkpoint_part1.safetensors`
4. Verify path: `/kaggle/input/model-checkpoint/checkpoint_part1.safetensors`

#### Configuration
```python
TRAINING_PART = 2
```

#### Run Training
1. Execute all cells
2. Training will automatically load Part 1 checkpoint
3. Continues training from epoch 19-36

#### Download Files
- `/kaggle/working/checkpoint_part2.safetensors`
- `/kaggle/working/processor_part2/`
- `/kaggle/working/metrics_part2.png`

---

### **Part 3: Continue Training**

#### Start Fresh Session
New Kaggle session with clean `/kaggle/working/`

#### Upload Checkpoint
Upload `checkpoint_part2.safetensors` to `/kaggle/input/model-checkpoint/`

#### Configuration
```python
TRAINING_PART = 3
```

#### Run & Download
- Execute all cells (epochs 37-54)
- Download `checkpoint_part3.safetensors`
- Download `processor_part3/`

---

### **Part 4: Final Training (Phase 1 + Phase 2)**

This part is longer (~4 hours) as it completes Phase 1 AND runs full Phase 2.

#### Start Fresh Session
New Kaggle session

#### Upload Checkpoint
Upload `checkpoint_part3.safetensors` to `/kaggle/input/model-checkpoint/`

#### Configuration
```python
TRAINING_PART = 4
```

#### What Happens
1. Completes Phase 1 (epochs 55-70)
2. Automatically transitions to Phase 2
3. Runs Phase 2 (all 7 epochs with lower learning rate)
4. This is the **final training model**

#### Download Files
- `/kaggle/working/checkpoint_final.safetensors` ← **MOST IMPORTANT**
- `/kaggle/working/processor_final/`
- `/kaggle/working/metrics_part4.png`

---

### **Part 5: Inference**

Generate predictions on test data.

#### Start Fresh Session
New Kaggle session (GPU recommended but not required)

#### Upload Final Checkpoint
Upload `checkpoint_final.safetensors` to `/kaggle/input/model-checkpoint/`

#### Configuration
```python
TRAINING_PART = 5
```

#### Run Inference
1. Execute all cells
2. Model loads final checkpoint
3. Processes all test audio files
4. Applies post-processing (normalization, Danda)

#### Download Final Output
- `/kaggle/working/submission.csv` ← **YOUR FINAL PREDICTIONS**

This CSV contains:
- `filename`: Test audio filename
- `transcription`: Bengali transcription with post-processing

---

## 📂 File Management Checklist

### What to Keep from Each Part

| Part | Must Keep | Optional |
|------|-----------|----------|
| 1 | ✅ `checkpoint_part1.safetensors` | `processor_part1/`, `metrics_part1.png` |
| 2 | ✅ `checkpoint_part2.safetensors` | `processor_part2/`, `metrics_part2.png` |
| 3 | ✅ `checkpoint_part3.safetensors` | `processor_part3/`, `metrics_part3.png` |
| 4 | ✅ `checkpoint_final.safetensors` | `processor_final/`, `metrics_part4.png` |
| 5 | ✅ `submission.csv` | - |

### Storage Requirements
- Each checkpoint: ~1.2 GB
- Each processor folder: ~50 KB
- Total storage needed: ~5-6 GB across all parts

---

## 🔧 Troubleshooting

### "File not found" Error

**Problem**: Checkpoint file not found
**Solution**: 
- Verify upload path: `/kaggle/input/model-checkpoint/`
- Check filename matches exactly: `checkpoint_part{N}.safetensors`
- Re-upload if necessary

### Out of Memory Error

**Problem**: GPU runs out of memory
**Solution**:
- Already optimized with `batch_size=1`
- Try T4 GPU instead of P100
- Restart kernel and try again

### Training Taking Too Long

**Problem**: Part exceeds 12 hours
**Solution**:
- Each part is designed for ~3-4 hours max
- If approaching 11 hours, stop early and create manual checkpoint
- Contact support if dataset is unexpectedly large

### Model Not Improving

**Problem**: WER not decreasing
**Solution**:
- Check first few parts - WER should decrease gradually
- Expected WER by Part 4: ~0.4-0.5
- Review preprocessing - ensure audio files are valid

### Upload Failed

**Problem**: Cannot upload checkpoint to Kaggle
**Solution**:
- Checkpoint size is ~1.2 GB (within Kaggle's 20GB limit)
- Use Kaggle's "Add Data" → "Upload" interface
- Ensure stable internet connection
- Try splitting upload if timeout occurs

---

## 📊 Expected Performance Metrics

### Phase 1 (Parts 1-4)

| Part | Epoch Range | Expected Training Loss | Expected WER |
|------|-------------|------------------------|--------------|
| 1 | 1-18 | 3.0 → 1.5 | 0.9 → 0.7 |
| 2 | 19-36 | 1.5 → 1.0 | 0.7 → 0.5 |
| 3 | 37-54 | 1.0 → 0.8 | 0.5 → 0.45 |
| 4 | 55-70 | 0.8 → 0.7 | 0.45 → 0.4 |

### Phase 2 (Part 4)

| Metric | Expected |
|--------|----------|
| Epochs | 7 |
| Learning Rate | 5e-6 (10x lower) |
| Final WER | ~0.4-0.5 |
| Purpose | Vocabulary exposure, minor improvements |

### Inference (Part 5)

| Metric | Expected |
|--------|----------|
| Processing Time | ~1 hour for full test set |
| Output Format | CSV with filename and transcription |
| Post-processing | Unicode normalization + Danda (।) |

---

## 🎓 Paper Reference

This implementation is based on:

> **Applying wav2vec2 for Speech Recognition on Bengali Common Voices Dataset**  
> Shahgir et al., arXiv:2209.06581

### Key Paper Details
- **Base Model**: `facebook/wav2vec2-large-xlsr-53`
- **Preprocessing**: 16kHz, silence removal, 1-10s clips
- **Phase 1**: 70 epochs, LR=5e-4, WD=2.5e-6
- **Phase 2**: 7 epochs, LR=5e-6, WD=2.5e-9
- **Post-processing**: Bengali normalization + Danda

---

## ✅ Verification Checklist

Use this to track your progress:

- [ ] Part 1 completed and checkpoint downloaded
- [ ] Part 2 completed and checkpoint downloaded
- [ ] Part 3 completed and checkpoint downloaded
- [ ] Part 4 completed and final checkpoint downloaded
- [ ] Part 5 completed and submission.csv downloaded
- [ ] Submission.csv has correct format (filename, transcription)
- [ ] Transcriptions contain Bengali Unicode characters
- [ ] Transcriptions end with Danda (।)

---

## 📞 Support

If you encounter issues:

1. **Check the notebook output**: Error messages are usually informative
2. **Review this guide**: Most issues are covered in Troubleshooting
3. **Verify file paths**: Most errors are due to incorrect upload paths
4. **Check Kaggle status**: Occasionally Kaggle has platform issues

---

## 🎉 Success!

Once you complete all 5 parts, you'll have:
- ✅ Fully trained wav2vec2 model for Bengali speech recognition
- ✅ Predictions on test set with post-processing
- ✅ Ready-to-submit CSV file
- ✅ Complete training logs and metrics

**Congratulations on completing the multi-part training pipeline!** 🚀
