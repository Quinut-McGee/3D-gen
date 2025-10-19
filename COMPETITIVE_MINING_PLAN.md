# 404-GEN Competitive Mining Implementation Plan

## 🚨 CRITICAL FINDINGS FROM DISCORD FAQ

The 404-GEN team explicitly states:
> "Our base miner code is NOT competitive"

This means everything we've optimized so far is **foundation work**, but we need a **complete pipeline overhaul** to truly compete.

---

## 📊 GAP ANALYSIS: Current vs Competitive

| Component | Current (Base Miner) | Required (Competitive) | Gap |
|-----------|---------------------|----------------------|-----|
| **Text-to-Image** | MVDream (multi-view diffusion) | Flux-Schnell or similar SOTA | ⚠️ MEDIUM |
| **Background Removal** | rembg | "Robust model, rembg not good enough" | ❌ CRITICAL |
| **3D Generation** | DreamGaussian (older) | Trellis or SOTA Gaussian Splat | ❌ CRITICAL |
| **Self-Validation** | None (just size check) | Pre-submit CLIP validation | ❌ CRITICAL |
| **Async Operation** | Sequential validator polling | Fully async multi-validator | ❌ CRITICAL |
| **Validator Management** | Basic selection | Blacklist WC validators (UID 180) | ⚠️ MEDIUM |
| **Speed** | 150s → 25s (optimized) | <5-15s competitive | ⚠️ MEDIUM |
| **Versatility** | Generic prompts | Wide variety optimization | ⚠️ MEDIUM |

**Priority Score**: 4 CRITICAL gaps, 3 MEDIUM gaps

---

## 🎯 COMPETITIVE MINING PIPELINE (Target Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                    ASYNC TASK MANAGER                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │Validator 1 │  │Validator 2 │  │Validator N │            │
│  │(Pull Task) │  │(Pull Task) │  │(Pull Task) │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │                │                │                    │
│        └────────────────┴────────────────┘                    │
│                         ▼                                     │
│              ┌──────────────────────┐                        │
│              │   TASK QUEUE POOL    │                        │
│              └──────────┬───────────┘                        │
└─────────────────────────┼────────────────────────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │    GENERATION PIPELINE (Per Task)   │
        │                                      │
        │  1. Text-to-Image (Flux-Schnell)    │
        │     └─> High-quality base image     │
        │                                      │
        │  2. Background Removal (SOTA)        │
        │     └─> Clean RGBA with alpha       │
        │                                      │
        │  3. 3D Generation (Trellis/SOTA)    │
        │     └─> Gaussian Splat output       │
        │                                      │
        │  4. Self-Validation (CLIP)          │
        │     └─> Score >= 0.6 threshold      │
        │         ├─ PASS → Submit            │
        │         └─ FAIL → Regenerate/Skip   │
        │                                      │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │    ASYNC SUBMISSION MANAGER         │
        │  (Don't wait for validator response)│
        └─────────────────────────────────────┘
```

**Key Improvements**:
1. ✅ Multiple validators polled simultaneously
2. ✅ Task queue allows parallel processing
3. ✅ SOTA models at each stage
4. ✅ Self-validation prevents penalties
5. ✅ Async submission for maximum throughput

---

## 🔧 IMPLEMENTATION ROADMAP

### **PHASE 1: Critical Infrastructure (Week 1)**
**Goal**: Replace critical bottlenecks, maintain current functionality

#### 1.1 Self-Validation System (HIGHEST PRIORITY)
**Current**: No validation, submitting low-quality results = cooldown penalties
**Target**: Pre-submit CLIP validation

**Implementation**:
```python
# New file: generation/validators/clip_validator.py

import torch
import clip
from PIL import Image

class CLIPValidator:
    def __init__(self, device="cuda", threshold=0.6):
        self.device = device
        self.threshold = threshold
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def validate(self, image_path: str, prompt: str) -> tuple[bool, float]:
        """
        Returns: (passes_threshold, clip_score)
        """
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([prompt]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Cosine similarity
            similarity = (image_features @ text_features.T).item()

        passes = similarity >= self.threshold
        return passes, similarity
```

**Integration** into `neurons/miner/workers.py`:
```python
from validators.clip_validator import CLIPValidator

validator = CLIPValidator(threshold=0.6)

# Before submission
render_preview = render_ply_to_image(results)  # New function needed
passes, score = validator.validate(render_preview, prompt)

if not passes:
    bt.logging.warning(f"Validation failed: {score:.3f} < 0.6, skipping submission")
    results = b""  # Submit empty
else:
    bt.logging.info(f"Validation passed: {score:.3f}")
```

**Expected Impact**:
- Eliminates cooldown penalties from low-quality submissions
- Increases effective throughput by 20-30%

**Timeline**: 2-3 days
**Complexity**: Medium
**Priority**: 🔴 CRITICAL

---

#### 1.2 Async Multi-Validator System
**Current**: Sequential validator polling (one at a time)
**Target**: Parallel task pulling from multiple validators

**Implementation**:
```python
# New file: neurons/miner/async_task_manager.py

import asyncio
from typing import List, Optional
import bittensor as bt

class AsyncTaskManager:
    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}

    async def pull_from_all_validators(
        self,
        validator_uids: List[int],
        metagraph: bt.metagraph,
        wallet: bt.wallet
    ):
        """Pull tasks from all validators simultaneously"""
        tasks = [
            self._pull_from_validator(uid, metagraph, wallet)
            for uid in validator_uids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for uid, result in zip(validator_uids, results):
            if isinstance(result, Exception):
                bt.logging.warning(f"Failed to pull from validator {uid}: {result}")
            elif result and result.task:
                await self.task_queue.put((uid, result.task))

    async def _pull_from_validator(self, uid, metagraph, wallet):
        """Pull task from single validator (non-blocking)"""
        async with bt.dendrite(wallet=wallet) as dendrite:
            return await dendrite.call(
                target_axon=metagraph.axons[uid],
                synapse=PullTask(),
                timeout=12.0
            )

    async def process_task_queue(self, generation_endpoints: List[str]):
        """Process tasks from queue using available endpoints"""
        workers = [
            self._task_worker(endpoint)
            for endpoint in generation_endpoints
        ]
        await asyncio.gather(*workers)

    async def _task_worker(self, endpoint: str):
        """Worker that processes tasks from queue"""
        while True:
            validator_uid, task = await self.task_queue.get()

            try:
                # Generate
                results = await generate(endpoint, task.prompt)

                # Validate
                if await self.validate_results(results, task.prompt):
                    # Submit (async, don't wait for response)
                    asyncio.create_task(
                        self.submit_results(validator_uid, task, results)
                    )
            except Exception as e:
                bt.logging.error(f"Task processing failed: {e}")
            finally:
                self.task_queue.task_done()
```

**Integration** into `neurons/miner/__init__.py`:
```python
async def run(self):
    task_manager = AsyncTaskManager(max_concurrent_tasks=4)

    while True:
        # Pull from ALL validators every 10s
        await task_manager.pull_from_all_validators(
            validator_uids=self.validator_selector.get_all_available(),
            metagraph=self.metagraph,
            wallet=self.wallet
        )

        # Process queue
        await task_manager.process_task_queue(self.config.generation.endpoints)

        await asyncio.sleep(10)
```

**Expected Impact**:
- 3-4x more tasks processed simultaneously
- No waiting between validator responses
- Better utilization of GPU

**Timeline**: 3-4 days
**Complexity**: High
**Priority**: 🔴 CRITICAL

---

#### 1.3 Validator Blacklisting
**Current**: Query all validators
**Target**: Skip WC validators (UID 180 mentioned)

**Implementation**:
```python
# In neurons/miner/config.py or validator_selector.py

BLACKLISTED_VALIDATORS = [180]  # WC validators to skip

class ValidatorSelector:
    def get_next_validator_to_query(self) -> int | None:
        # ... existing code ...

        if self._next_uid in BLACKLISTED_VALIDATORS:
            bt.logging.debug(f"Skipping blacklisted validator [{self._next_uid}]")
            self._next_uid = (self._next_uid + 1) % metagraph.n
            continue  # Skip this validator

        # ... rest of logic ...
```

**Expected Impact**: Saves wasted compute on non-rewarding validators
**Timeline**: 1 day
**Complexity**: Low
**Priority**: 🟡 MEDIUM

---

### **PHASE 2: Model Upgrades (Week 2-3)**
**Goal**: Replace with SOTA models

#### 2.1 Text-to-Image: Flux-Schnell Integration

**Why Flux-Schnell**:
- State-of-the-art quality (2024)
- Fast inference (schnell = fast in German)
- Commercial use allowed
- Better than MVDream for diverse prompts

**Implementation**:
```python
# New file: generation/models/flux_generator.py

from diffusers import FluxPipeline
import torch

class FluxImageGenerator:
    def __init__(self, device="cuda"):
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        ).to(device)

    def generate(self, prompt: str, steps: int = 4) -> Image:
        """
        Generate image from text prompt

        Args:
            prompt: Text description
            steps: Inference steps (4 is optimal for schnell)

        Returns:
            PIL Image
        """
        return self.pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Schnell works best without guidance
            height=512,
            width=512
        ).images[0]
```

**Expected Impact**:
- Better prompt understanding
- Faster than MVDream (4 steps vs 50+)
- More versatile across object categories

**Timeline**: 4-5 days (includes testing)
**Complexity**: Medium
**Priority**: 🔴 CRITICAL

---

#### 2.2 Background Removal: SOTA Upgrade

**Current**: rembg (FAQ says "not good enough")

**Options**:
1. **BRIA RMBG 2.0** (recommended)
   - SOTA accuracy (2024)
   - Fast inference
   - Commercial license available

2. **InSPyReNet**
   - Very high quality
   - Slower but better edges

**Implementation**:
```python
# New file: generation/models/background_remover.py

from transformers import AutoModelForImageSegmentation
import torch
from PIL import Image
import numpy as np

class SOTABackgroundRemover:
    def __init__(self, device="cuda"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True
        ).to(device)
        self.device = device

    def remove_background(self, image: Image) -> Image:
        """
        Remove background with SOTA model

        Returns:
            RGBA image with transparent background
        """
        # Preprocess
        input_images = self.model.preprocess(image).to(self.device)

        # Predict
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()

        # Post-process
        pred = preds[0].squeeze()
        mask = (pred.numpy() * 255).astype(np.uint8)

        # Create RGBA
        rgba = image.convert("RGB")
        rgba.putalpha(Image.fromarray(mask))

        return rgba
```

**Expected Impact**:
- Better edge quality
- Fewer artifacts
- Higher CLIP scores

**Timeline**: 2-3 days
**Complexity**: Low-Medium
**Priority**: 🟡 MEDIUM

---

#### 2.3 3D Generation: Trellis or SOTA Alternative

**Current**: DreamGaussian (2023, outdated)
**Target**: Trellis (2024, SOTA)

**Why Trellis**:
- Released Nov 2024 by Microsoft
- State-of-the-art Gaussian Splat generation
- Fast inference
- Better quality than DreamGaussian

**Implementation**:
```python
# New file: generation/models/trellis_generator.py

from trellis import TrellisGenerator  # Hypothetical API
import torch

class Trellis3DGenerator:
    def __init__(self, device="cuda"):
        self.model = TrellisGenerator.from_pretrained(
            "microsoft/trellis-base"
        ).to(device)

    def generate_gaussian_splat(
        self,
        image: Image,
        prompt: str
    ) -> bytes:
        """
        Generate Gaussian Splat from image + prompt

        Returns:
            PLY file bytes
        """
        with torch.no_grad():
            splat = self.model(
                image=image,
                text=prompt,
                output_format="gaussian_splat"
            )

        return splat.to_ply()
```

**Alternative: InstantMesh → Gaussian Splat Conversion**
If Trellis integration is difficult:

```python
# Use InstantMesh for mesh, convert to Gaussian Splat
from instant_mesh import InstantMeshGenerator
from mesh_to_gaussian import convert_mesh_to_splat

mesh = InstantMeshGenerator().generate(image, prompt)
splat = convert_mesh_to_splat(mesh)
```

**Expected Impact**:
- 2-3x better quality
- Potentially faster
- More consistent results

**Timeline**: 5-7 days (most complex)
**Complexity**: High
**Priority**: 🔴 CRITICAL

---

### **PHASE 3: Optimization & Scaling (Week 4+)**
**Goal**: Maximum performance from hardware

#### 3.1 Multi-GPU Scaling

**Your Setup**: 1x RTX 4090 (24GB)
**Potential**: Could add more GPUs if needed

**Implementation**:
```python
# Load balance across GPUs
import torch.multiprocessing as mp

def run_generation_endpoint(gpu_id: int, port: int):
    torch.cuda.set_device(gpu_id)

    # Initialize models on specific GPU
    flux = FluxImageGenerator(device=f"cuda:{gpu_id}")
    trellis = Trellis3DGenerator(device=f"cuda:{gpu_id}")

    # Run FastAPI server
    app = create_app(flux, trellis)
    uvicorn.run(app, port=port)

if __name__ == "__main__":
    # Spawn processes for each GPU
    processes = []
    for gpu_id in range(torch.cuda.device_count()):
        p = mp.Process(
            target=run_generation_endpoint,
            args=(gpu_id, 10006 + gpu_id)
        )
        p.start()
        processes.append(p)
```

**Expected Impact**: Linear scaling with GPU count
**Timeline**: 2-3 days
**Complexity**: Medium
**Priority**: 🟢 LOW (only if budget allows)

---

#### 3.2 Prompt Category Optimization

**FAQ Says**: "It's not enough to master generating cute animals"

**Implementation**:
```python
# New file: generation/prompt_optimizer.py

CATEGORY_OPTIMIZATIONS = {
    "animals": {
        "flux_steps": 4,
        "trellis_detail": "high",
        "background_threshold": 0.95
    },
    "vehicles": {
        "flux_steps": 6,  # More steps for hard edges
        "trellis_detail": "medium",
        "background_threshold": 0.90
    },
    "architecture": {
        "flux_steps": 8,
        "trellis_detail": "high",
        "background_threshold": 0.85
    },
    # ... more categories
}

def detect_category(prompt: str) -> str:
    """Use classifier to detect prompt category"""
    # Could use simple keyword matching or a classifier model
    pass

def optimize_for_category(prompt: str) -> dict:
    """Return category-specific parameters"""
    category = detect_category(prompt)
    return CATEGORY_OPTIMIZATIONS.get(category, DEFAULT_PARAMS)
```

**Expected Impact**:
- Better versatility across prompts
- Higher average CLIP scores
- Better ELO performance

**Timeline**: 3-4 days
**Complexity**: Medium
**Priority**: 🟡 MEDIUM

---

## 📅 IMPLEMENTATION TIMELINE

### **Week 1: Critical Foundation**
```
Day 1-2:  Self-validation system (CLIP pre-check)
Day 3-4:  Async multi-validator polling
Day 5:    Validator blacklisting
Day 6-7:  Testing & integration
```

**Deliverable**: Miner with self-validation and async operation
**Expected Improvement**: 2-3x throughput, eliminate cooldown penalties

---

### **Week 2-3: Model Upgrades**
```
Day 8-10:  Flux-Schnell text-to-image integration
Day 11-12: SOTA background removal
Day 13-18: Trellis 3D generation (most complex)
Day 19-21: End-to-end testing
```

**Deliverable**: Full SOTA pipeline
**Expected Improvement**: 5-10x quality, competitive with top miners

---

### **Week 4+: Optimization**
```
Day 22-24: Prompt category optimization
Day 25-27: Multi-GPU scaling (if needed)
Day 28-30: Performance tuning
```

**Deliverable**: Production-ready competitive miner
**Expected Improvement**: Top 10-20% of network

---

## 💰 COST-BENEFIT ANALYSIS

### **Costs**:

**Development Time**: 3-4 weeks full implementation
**Model Storage**: ~50-100 GB additional disk space
**Potential GPU Upgrade**: Current 4090 may be sufficient

### **Benefits**:

**Current Rewards** (with optimizations): ~30 points per 4h
**Competitive Rewards** (with SOTA pipeline): ~80-120 points per 4h

**ROI**: 3-4x reward increase
**Break-even**: 1-2 weeks after deployment

---

## 🎯 PHASED DEPLOYMENT STRATEGY

### **Phase 1A: Quick Wins (Deploy This Week)**
**Focus**: Self-validation only

```python
# Minimal implementation in workers.py
from clip_validator import CLIPValidator
validator = CLIPValidator()

# Before submission:
if not validator.validate(results, prompt):
    results = b""  # Skip low-quality
```

**Impact**: +30-50% effective throughput
**Risk**: Low
**Time**: 2-3 days

---

### **Phase 1B: Async Operations (Week 2)**
**Focus**: Multi-validator polling

**Impact**: +200-300% throughput
**Risk**: Medium (code complexity)
**Time**: 4-5 days

---

### **Phase 2: Model Swap (Week 3-4)**
**Focus**: Replace models one at a time

**Order**:
1. Flux-Schnell (easiest, big impact)
2. Background removal (medium difficulty)
3. Trellis (hardest, biggest impact)

**Impact**: +300-500% quality
**Risk**: High (new models)
**Time**: 10-14 days

---

## 🚦 DECISION POINTS

### **Option A: Full Rewrite (Recommended)**
- Implement entire SOTA pipeline
- Timeline: 3-4 weeks
- Expected final position: Top 10-20%
- **Best for**: Long-term competitive mining

### **Option B: Incremental Upgrades**
- Deploy self-validation immediately
- Add async next week
- Model upgrades month 2
- Timeline: 6-8 weeks
- Expected final position: Top 30-40%
- **Best for**: Risk-averse approach

### **Option C: Hybrid Approach (RECOMMENDED)**
- Week 1: Self-validation + async (critical infrastructure)
- Week 2-3: Test with current models, measure improvement
- Week 4+: Add SOTA models only if needed to compete
- Timeline: 4-6 weeks
- **Best for**: Data-driven optimization

---

## 🔬 RESEARCH TASKS

Before implementing models, we need to research:

### **Flux-Schnell**
- [ ] Verify commercial license
- [ ] Test inference speed on 4090
- [ ] Benchmark quality vs MVDream
- [ ] Check VRAM requirements

### **Trellis**
- [ ] Find official repo/implementation
- [ ] Check if it outputs Gaussian Splats directly
- [ ] Test inference time
- [ ] Evaluate quality on test prompts

### **BRIA RMBG 2.0**
- [ ] Verify commercial license
- [ ] Test speed vs rembg
- [ ] Benchmark edge quality
- [ ] Integration complexity

---

## 📊 SUCCESS METRICS

### **Week 1 (After Self-Validation)**
- ✅ Zero cooldown penalties
- ✅ 70%+ submission acceptance
- ✅ 2x throughput vs baseline

### **Week 2 (After Async)**
- ✅ 4+ concurrent tasks
- ✅ 4x throughput vs baseline
- ✅ <10% idle time

### **Week 4 (After Model Upgrades)**
- ✅ Average CLIP > 0.75
- ✅ Top 25% ELO ranking
- ✅ Consistent winning duels

### **Week 6 (Fully Optimized)**
- ✅ Top 10-20% rewards
- ✅ Average CLIP > 0.80
- ✅ <10s generation time
- ✅ 95%+ uptime

---

## 🎬 IMMEDIATE NEXT STEPS

### **Step 1: Decide on Approach**
Choose: Full Rewrite, Incremental, or Hybrid?

### **Step 2: Research Phase (1-2 days)**
- Investigate Trellis availability
- Test Flux-Schnell locally
- Verify all licenses

### **Step 3: Start with Self-Validation (2-3 days)**
- Lowest risk
- Immediate benefit
- Can deploy while researching models

### **Step 4: Implement Async (4-5 days)**
- High complexity but huge impact
- Necessary foundation for scaling

### **Step 5: Model Integration (2-3 weeks)**
- One model at a time
- Test thoroughly before next

---

## ❓ QUESTIONS TO ANSWER

1. **Budget**: Can you afford additional GPUs if needed?
2. **Timeline**: Do you have 3-4 weeks for full implementation?
3. **Risk Tolerance**: Prefer incremental or big-bang deployment?
4. **Technical Support**: Is there a team or just you?
5. **Current Performance**: What are your actual rewards/day currently?

---

## 📝 RECOMMENDATION

**My recommendation: Hybrid Approach**

**Week 1**:
1. Deploy self-validation (immediate benefit, low risk)
2. Research Trellis, Flux-Schnell availability

**Week 2**:
1. Implement async multi-validator
2. Test with current models
3. Measure improvement

**Week 3-4**:
1. If still not competitive → integrate Flux + Trellis
2. If competitive → optimize existing pipeline

**Reasoning**:
- Gets quick wins deployed fast
- Validates architecture before big model changes
- Data-driven decision on expensive model integration
- Lower risk than full rewrite

---

**Ready to proceed? Let me know which approach you prefer and I'll start implementing!** 🚀
