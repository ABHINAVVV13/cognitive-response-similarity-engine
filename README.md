<div align="center">

# Cognitive Response Similarity Engine (CRSE)

**Predict neural response pattern similarity between video stimuli.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TRIBE v2](https://img.shields.io/badge/Powered%20by-TRIBE%20v2-purple.svg)](https://github.com/facebookresearch/tribev2)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Weights-yellow.svg)](https://huggingface.co/facebook/tribev2)

</div>

---

## Overview

CRSE utilizes [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2), a trimodal brain encoding model, to predict fMRI brain responses to naturalistic stimuli. The engine processes two videos through TRIBE v2 to obtain predicted cortical activation maps (approximately 20,000 vertices on the fsaverage5 mesh). It then computes the statistical similarity of these neural response patterns, focusing on anatomical brain regions responsible for:

- **Emotional Processing**: Insula, cingulate cortex, temporal pole, orbitofrontal cortex.
- **Visual Processing**: Calcarine sulcus (V1), cuneus, fusiform gyrus, lateral occipital complex.
- **Auditory Processing**: Heschl's gyrus (A1), planum temporale.
- **Language Comprehension**: Broca's area, Wernicke's area, angular gyrus.
- **Default Mode Network**: Precuneus, medial PFC, temporoparietal junction (TPJ).
- **Motor and Somatosensory**: Precentral and postcentral gyri.
- **Executive Control**: Dorsolateral and medial prefrontal cortex.

---

## Architecture

```text
┌─────────────┐     ┌──────────────────────────────────────┐     ┌──────────────────┐
│             │     │          Meta TRIBE v2               │     │                  │
│  Video A    │────▶│  V-JEPA2 + Wav2Vec-BERT + LLaMA 3.2  │────▶│  Brain Map A     │
│  (.mp4)     │     │  → Transformer → Cortical Mapping    │     │  (T×20k verts)   │
│             │     │                                      │     │                  │
└─────────────┘     └──────────────────────────────────────┘     └────────┬─────────┘
                                                                          │
                                                                          ▼
                                                                 ┌────────────────────┐
                                                                 │  CRSE Comparator   │
                                                                 │                    │
                                                                 │  • Cosine sim      │
                                                                 │  • Pearson corr    │
                                                                 │  • Temporal ISC    │
                                                                 │  • RSA             │
                                                                 │  • Per-region      │
                                                                 └────────────────────┘
                                                                          │
┌─────────────┐     ┌──────────────────────────────────────┐              ▼
│             │     │          Meta TRIBE v2               │     ┌────────────────────┐
│  Video B    │────▶│  V-JEPA2 + Wav2Vec-BERT + LLaMA 3.2  │────▶│  Brain Map B       │
│  (.mp4)     │     │  → Transformer → Cortical Mapping    │     │  (T×20k verts)     │
│             │     │                                      │     │                    │
└─────────────┘     └──────────────────────────────────────┘     └────────────────────┘
```

---

## Quick Start

```python
from crse import CRSEngine

# Initialize (downloads TRIBE v2 weights on first run)
engine = CRSEngine()

# Compare two videos
result = engine.compare("video_a.mp4", "video_b.mp4")

# Print formatted results
print(result.summary())

# Access specific regions
for region in result.regions:
    print(f"{region.name}: cosine={region.metrics['cosine_similarity']:.3f}")

# Save to JSON
result.save("results.json")
```

---

## Installation

### Prerequisites

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager). Install via `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **PyTorch ≥ 2.5** (CUDA recommended for GPU acceleration)
- **FFmpeg** (For video/audio processing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/ABHINAVVV13/cognitive-response-similarity-engine.git
cd cognitive-response-similarity-engine
```

### Step 2: Create Environment and Install Dependencies

```bash
# Install PyTorch first (CUDA 12.4 is recommended for GPU environments)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only alternative:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install TRIBE v2

```bash
git clone https://github.com/facebookresearch/tribev2.git
uv pip install -e ./tribev2
```

### Step 4: Install CRSE

```bash
# Core installation
uv pip install -e .

# With Streamlit dashboard
uv pip install -e ".[app]"

# With development tools (pytest, ruff)
uv pip install -e ".[dev]"

# Or install all options
uv pip install -e ".[app,dev,runpod]"
```

### Step 5: Download Model Weights

TRIBE v2 model weights are hosted on HuggingFace and will be downloaded on first use. Note that acceptance of the model license is required.

1. Navigate to [https://huggingface.co/facebook/tribev2](https://huggingface.co/facebook/tribev2).
2. Sign in to your HuggingFace account.
3. Accept the CC-BY-NC-4.0 model license.
4. Log in via CLI:
   ```bash
   uv pip install huggingface_hub
   huggingface-cli login
   ```

The model checkpoint (approximately 2 GB) is cached locally after the initial download.

### Step 6: Install spaCy Language Model

```bash
uv run python -m spacy download en_core_web_sm
```

---

## Usage

### Command-Line Interface

```bash
# Local execution
crse compare video_a.mp4 video_b.mp4

# Save results and export plots
crse compare video_a.mp4 video_b.mp4 --output results.json --plot

# Specify compute device and anatomical regions
crse compare video_a.mp4 video_b.mp4 --device cuda --regions emotional_limbic visual_cortex

# Remote execution on RunPod serverless GPU
crse compare --runpod https://example.com/video_a.mp4 https://example.com/video_b.mp4

# List defined brain regions
crse regions

# Verify RunPod endpoint status
crse runpod health
```

### Python API (Local)

```python
from crse import CRSEngine

engine = CRSEngine(
    model_id="facebook/tribev2",
    cache_folder="./cache",
    device="auto",             # Select "auto", "cuda", or "cpu"
    regions=None,              # Defaults to all anatomical regions
)

result = engine.compare("speech.mp4", "music_video.mp4")

# Whole-brain data
print(result.whole_brain)

# Regional data
for region in result.regions:
    print(f"{region.name}: {region.mean_score:.3f}")
    print(f"  {region.description}")

result.save("output.json")
```

### Python API (RunPod Cloud GPU)

```python
from crse.runpod_client import CRSERunPodClient

client = CRSERunPodClient(
    api_key="your_runpod_api_key",      
    endpoint_id="your_endpoint_id",     
)

# Synchronous request
result = client.compare(
    video_a_url="https://example.com/video_a.mp4",
    video_b_url="https://example.com/video_b.mp4",
    regions=["emotional_limbic", "visual_cortex"],
)
print(result["whole_brain"])

# Asynchronous request
job_id = client.compare_async(
    video_a_url="https://example.com/video_a.mp4",
    video_b_url="https://example.com/video_b.mp4",
)
result = client.get_result(job_id)
```

### Streamlit Dashboard

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

---

## Similarity Metrics

| Metric | Functional Description | Range |
|--------|----------------------|-------|
| **Cosine Similarity** | Cosine angle calculated between mean spatial activation vectors. | [-1, 1] |
| **Pearson Correlation** | Linear relationship between the mean spatial activation patterns. | [-1, 1] |
| **Spatial Pattern** | Pearson correlation of mean-centered spatial patterns. Isolates regional variation from global magnitude. | [-1, 1] |
| **Temporal Correlation** | Average temporal Pearson correlation across all individual vertices. | [-1, 1] |
| **Temporal ISC** | Inter-Stimulus Correlation assessing global, spatially-averaged signal changes over time. | [-1, 1] |
| **RSA** | Representational Similarity Analysis computed via Spearman correlation of Temporal Representational Dissimilarity Matrices. | [-1, 1] |

**General Interpretation Guidelines:**
- **> 0.7**: High structural and functional similarity.
- **0.3 to 0.7**: Moderate correlation in response patterns.
- **0.0 to 0.3**: Weak or minimal similarity.
- **< 0.0**: Dissimilar or potentially opposing network activations.

---

## Brain Regions

CRSE utilizes the Destrieux anatomical atlas (2009) via the `nilearn` package to parcellate the fsaverage5 cortical mesh into specific regions of interest (ROIs):

| Anatomical Region Group | Included Structures | Associated Function |
|--------|----------------|-------------------|
| `emotional_limbic` | Insula, cingulate, temporal pole, OFC. | Affective processing, interoception, valuation. |
| `visual_cortex` | V1, cuneus, fusiform, lateral occipital complex. | Visual perception and object recognition. |
| `auditory_cortex` | Heschl's gyrus, planum temporale. | Sound processing and pitch differentiation. |
| `language_network` | Broca's area, Wernicke's area, angular gyrus. | Speech production and semantic comprehension. |
| `motor_somatosensory` | Precentral gyrus, postcentral gyrus. | Voluntary motor control and tactile feedback. |
| `prefrontal_executive` | DLPFC, mPFC, frontopolar cortex. | Working memory, cognitive control, spatial reasoning. |
| `default_mode_network` | Precuneus, IPL, mPFC, TPJ. | Self-referential cognition and theory of mind. |

---

## RunPod Serverless Deployment

CRSE includes a RunPod serverless handler to facilitate remote GPU execution for resource-intensive inference. 

1. Ensure RunPod CLI/account is configured.
2. Build the provided Docker container image:
```bash
docker build -t crse-worker:latest -f runpod/Dockerfile .
```
3. Push to your registry and configure a Serverless endpoint. 
4. The endpoint expects HTTPS JSON payloads and securely returns predictions from the active worker. See the Python API section for interaction examples.

---

## License

### CRSE Code (MIT License)

The source code for CRSE is distributed under the [MIT License](LICENSE). 

### TRIBE v2 Model (CC-BY-NC-4.0)

The TRIBE v2 model weights are distributed by Meta strictly under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) license. 
- Academic and educational use is permitted.
- Commercial usage is prohibited without secondary authorization from Meta.

---

## Citation

Appropriate citation is requested if this software features in academic work:

```bibtex
@software{crse2026,
  title={Cognitive Response Similarity Engine},
  author={Putta, Abhinav},
  year={2026},
  url={https://github.com/ABHINAVVV13/cognitive-response-similarity-engine}
}

@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stéphane and Rapin, Jérémy and Benchetrit, Yohann and Brookes, Teon and Begany, Katelyn and Raugel, Joséphine and Banville, Hubert and King, Jean-Rémi},
  year={2026}
}
```

---

<div align="center">
Developed by Abhinav Putta.
</div>
