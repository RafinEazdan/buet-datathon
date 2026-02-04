# 🚀 Quick Reference Card

## Part-by-Part Configuration

### Part 1
```python
TRAINING_PART = 1
```
- **Epochs**: 1-18
- **Duration**: ~3 hours
- **Download**: `checkpoint_part1.safetensors`

---

### Part 2
```python
TRAINING_PART = 2
```
- **Upload**: `checkpoint_part1.safetensors` → `/kaggle/input/model-checkpoint/`
- **Epochs**: 19-36
- **Duration**: ~3 hours
- **Download**: `checkpoint_part2.safetensors`

---

### Part 3
```python
TRAINING_PART = 3
```
- **Upload**: `checkpoint_part2.safetensors` → `/kaggle/input/model-checkpoint/`
- **Epochs**: 37-54
- **Duration**: ~3 hours
- **Download**: `checkpoint_part3.safetensors`

---

### Part 4
```python
TRAINING_PART = 4
```
- **Upload**: `checkpoint_part3.safetensors` → `/kaggle/input/model-checkpoint/`
- **Epochs**: 55-70 (Phase 1) + 1-7 (Phase 2)
- **Duration**: ~4 hours
- **Download**: `checkpoint_final.safetensors`

---

### Part 5
```python
TRAINING_PART = 5
```
- **Upload**: `checkpoint_final.safetensors` → `/kaggle/input/model-checkpoint/`
- **Duration**: ~1 hour
- **Download**: `submission.csv`

---

## Common Commands

### Check Current Part
Look for this in the notebook output:
```
====================================
TRAINING PART X SELECTED
====================================
```

### Download Files from Kaggle
1. Navigate to `/kaggle/working/` in Kaggle's file explorer
2. Right-click on file → Download
3. Or use: `!cp /kaggle/working/*.safetensors .`

### Verify Upload
```python
import os
print(os.listdir('/kaggle/input/model-checkpoint/'))
```
Should show your uploaded checkpoint file.

---

## File Paths Reference

| Description | Path |
|-------------|------|
| Upload checkpoint here | `/kaggle/input/model-checkpoint/` |
| Downloaded files location | `/kaggle/working/` |
| Training data (audio) | `/kaggle/input/.../train/audio/` |
| Training data (text) | `/kaggle/input/.../train/annotation/` |
| Test data | `/kaggle/input/.../test/` |

---

## Error Quick Fixes

| Error | Fix |
|-------|-----|
| "File not found" | Check upload path is `/kaggle/input/model-checkpoint/` |
| "Out of memory" | Restart kernel, already using batch_size=1 |
| "Session timeout" | Each part should be < 12 hours, start fresh |
| "Checkpoint mismatch" | Verify correct part number uploaded |

---

## Expected WER by Part

- **Part 1**: ~0.7
- **Part 2**: ~0.5
- **Part 3**: ~0.45
- **Part 4**: ~0.4
- **Final**: ~0.4-0.5

---

## Checkpoint Files

| File | Size | Purpose |
|------|------|---------|
| `checkpoint_part1.safetensors` | ~1.2 GB | After 18 epochs |
| `checkpoint_part2.safetensors` | ~1.2 GB | After 36 epochs |
| `checkpoint_part3.safetensors` | ~1.2 GB | After 54 epochs |
| `checkpoint_final.safetensors` | ~1.2 GB | Final trained model |
| `submission.csv` | ~100 KB | Final predictions |

---

## Session Workflow

```
START → Set TRAINING_PART → Run All Cells → Download Checkpoint → END
  ↓
NEW SESSION → Upload Previous Checkpoint → Repeat
```

---

## Key Points

✅ **Always start fresh session** for each part  
✅ **Upload to correct path**: `/kaggle/input/model-checkpoint/`  
✅ **Enable GPU** for all parts (optional for Part 5)  
✅ **Download checkpoint** before session expires  
✅ **Verify TRAINING_PART** is set correctly  

---

## Emergency: Session About to Expire

If approaching 12 hours:

1. **Stop training** (interrupt kernel)
2. **Save emergency checkpoint**:
   ```python
   from safetensors.torch import save_file
   save_file(model.state_dict(), '/kaggle/working/emergency_checkpoint.safetensors')
   ```
3. **Download immediately**
4. **Continue in next session**

---

Print this card and keep it handy! 📄
