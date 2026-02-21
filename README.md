# AReS: Alternative Reprogramming for Service Models

<p align="center">
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/CVPR-2026-0073AE.svg" alt="CVPR 2026"></a>
  <a href="https://github.com/yunbeizhang/AReS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Code-Coming_Soon-yellow.svg" alt="Code Coming Soon"></a>
</p>

<p align="center">
  <strong>Yunbei Zhang<sup>1</sup></strong> &emsp;
  <strong>Chengyi Cai<sup>2</sup></strong> &emsp;
  <strong>Feng Liu<sup>2</sup></strong> &emsp;
  <strong>Jihun Hamm<sup>1</sup></strong>
</p>
<p align="center">
  <sup>1</sup>Tulane University &emsp; <sup>2</sup>University of Melbourne
</p>

<p align="center">
  <a href="Alternative_Reprogramming_for_Service_Models.pdf"><strong>[Paper]</strong></a>
</p>

---

## Overview

**AReS** proposes an alternative to the conventional zeroth-order optimization (ZOO) paradigm for adapting closed-box service models (APIs) to downstream tasks. Instead of costly, continuous API queries, AReS performs a **single-pass interaction** with the service API to prime a local pre-trained encoder, then conducts all subsequent adaptation and inference **entirely locally** — eliminating further API costs.

<p align="center">
  <img src="assets/workflow.png" width="100%">
</p>

**(a)** Previous closed-box methods use ZOO, requiring numerous API calls during training and one per image at inference. **(b)** AReS performs a one-time priming to prepare a local model, enabling efficient glass-box reprogramming with no further API dependency.

## Highlights

- **Effective on modern APIs:** On GPT-4o, AReS achieves **+27.8%** over zero-shot, where ZOO-based methods provide little to no improvement.
- **State-of-the-art accuracy:** Outperforms prior methods by **+2.5%** (VLMs) and **+15.6%** (VMs) on average across 10 datasets.
- **99.99% fewer API calls:** Reduces API calls from ~10<sup>8</sup> to ~10<sup>3</sup>, and training time from 32+ hours to under 30 minutes.
- **Cost-free inference:** Once primed, all inference runs locally with zero API cost.

<p align="center">
  <img src="assets/gpt4o.png" width="32%">
  <img src="assets/api_comparison.png" width="32%">
  <img src="assets/time_comparison.png" width="32%">
</p>

**(a)** On GPT-4o, ZOO-based methods show limited effectiveness while incurring high costs. **(b, c)** On CLIP ViT-B/16 (Flowers102), AReS uses only ~10<sup>3</sup> API calls and 0.4 hours vs. ~10<sup>8</sup> calls and 32+ hours for prior methods.

## Code

Code will be released soon. Stay tuned!

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{zhang2026ares,
  title={Alternative Reprogramming for Service Models},
  author={Zhang, Yunbei and Cai, Chengyi and Liu, Feng and Hamm, Jihun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

This work was supported in part by NSF. We thank the reviewers for their constructive feedback.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
