<div align="center">
<div>
<a href="https://github.com/OBI-Future/PictOBI-20k"><img src="https://visitor-badge.laobi.icu/badge?page_id=OBI-Future/PictOBI-20k"/></a>
    <a href="https://github.com/OBI-Future/PictOBI-20k"><img src="https://img.shields.io/github/stars/OBI-Future/PictOBI-20k"/></a>
    <a href="https://arxiv.org/abs/2509.05773"><img src="https://img.shields.io/badge/Arxiv-2509.05773-red"/></a>
    <a href="https://github.com/OBI-Future/PictOBI-20k"><img src="https://img.shields.io/badge/Awesome-PictOBI--20k-orange"/></a>
</div>

<h1>PictOBI-20k: Unveiling Large Multimodal Models in Visual Decipherment for Pictographic Oracle Bone Characters 🔍</h1>

_A pioneering benchmark bridging large multimodal models with the visual decipherment of ancient Chinese oracle bone scripts_


<div>
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=NSR4UkMAAAAJ" target="_blank">Zijian Chen</a><sup>1,2,†</sup>,  
    Wenjie Hua<sup>3,†</sup>,  
    Jinhao Li<sup>4</sup>,  
    Lirong Deng<sup>5</sup>,  
    Fan Du<sup>6</sup>,  
    <a href="https://shss.sjtu.edu.cn/Web/FacultyDetail/46?f=1&t=4" target="_blank">Tingzhu Chen</a><sup>1,★</sup>,  
    <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=E6zbSYgAAAAJ" target="_blank">Guangtao Zhai</a><sup>1,2,★</sup>
</div>


<div>
  <sup>1</sup>Shanghai Jiao Tong University  
  <sup>2</sup>Shanghai AI Lab  
  <sup>3</sup>Wuhan University  
  <sup>4</sup>East China Normal University  
  <sup>5</sup>Macao Polytechnic University  
  <sup>6</sup>Southern University of Science and Technology  

  <sup>†</sup>Equal contribution &nbsp;&nbsp; <sup>★</sup>Corresponding authors

</div>
</div>

<p align="center">
  <img src="assets/intro.jpg" width="80%">
</p>
  
> Overview of **PictOBI-20k**: We present **PictOBI-20k**, a large-scale dataset for evaluating LMMs on the visual decipherment of pictographic Oracle Bone Characters (OBCs). The dataset comprises **20k carefully curated OBC–object image pairs** and over **15k multi-choice questions**. To further assess visual reasoning, we provide **subjective annotations** examining the consistency of reference points between humans and LMMs. Experimental results suggest that while general LMMs exhibit preliminary visual decipherment ability, they often fail to effectively leverage visual information and remain constrained by language priors. We hope PictOBI-20k can serve as a foundation for advancing evaluation and optimization of **visual attention in OBC-oriented LMMs**.

Examples of reference point heatmaps:

<p align="center">
  <img src="assets/reference_map.jpg" width="100%">
</p>


## Release
- [TBA] Public datasets release.  
- [2025/09/09] 🔥 Github repository for **PictOBI-20k** is online.


## Benchmark Design
##### Focusing on the Visual-Decipherment Abilities of LMMs for OBCs

We collect OBC and real object images from 12 sources, covering multiple font appearances and categories. Based on these, we construct 15,175 multi-choice questions for LMM evaluation. Meanwhile, we conduct human annotations for obtaining reference points on OBC-object image pairs.

<div style="width: 80%; text-align: center; margin:auto;">
      <img style="width:100%" src="assets/overview.jpg">
</div>

## OBC and Real-Object Image Sources

We collect OBC images from **3 OBC-centric** ancient script websites, YinQiWenYuan, XiaoXueTang, and GuoXueDaShi, as well as **5 open-source OBC datasets**, including **Oracle-241, Oracle-50k, HUST-OBS, OBI125, and OBIdatasetIJDH**. Corresponding **real-object images** (≈4.8k) are carefully collected from Freepik, Pexels, Pinterest, and the Academia Sinica Bronze Ware Database.

<p align="center">
  <img src="assets/source.png" width="60%">
</p>

## Benchmark Candidates

We evaluate **11** LMMs—including GPT-4o, Gemini 2.5 Pro, Claude 4 Sonnet, GLM-4.5V, the Qwen2.5-VL family, and the InternVL3 series—alongside three vision encoders (DINOv2-L/14, CLIP-L/14, InternViT-300M) to assess multimodal and visual-only performance on pictographic OBCs.

<p align="center">
  <img src="assets/candidate.jpg" width="60%">
</p>


## Performance Benchmark on Pictographic OBC Tasks 

<details close>
<summary>Results on the classification tasks — Average accuracy in terms of OBC classes (click to expand)</summary>

<p align="center">
  <img src="assets/accuracy-class.jpg" width="80%">
</p>
</details>

<details close>
<summary>Results on the consistency tasks — The consistency (%) of visual reference on 240 OBC–object pairs between humans and LMMs (click to expand)</summary>

<p align="center">
  <img src="assets/consistency.jpg" width="80%">
</p>
</details>

<details close>
<summary>Results on the vision encoder analysis — Visualization of attention maps (left) and accuracy from direct readout of visual encoders (right) (click to expand)</summary>

<p align="center">
  <img src="assets/vision_backbone.jpg" width="80%">
</p>
</details>


## Dataset 📦
- [x] To be released with `dataset/` OBC and real-object image

## Contact 📧
Please contact the authors for queries.

- Zijian Chen, `zijian.chen@sjtu.edu.cn`

## Citation📎
If you find our work interesting, please feel free to cite our paper:
```
@article{chen2025pictobi20kunveilinglargemultimodal,
      title={PictOBI-20k: Unveiling Large Multimodal Models in Visual Decipherment for Pictographic Oracle Bone Characters}, 
      author={Zijian Chen and Wenjie Hua and Jinhao Li and Lirong Deng and Fan Du and Tingzhu Chen and Guangtao Zhai},
      journal={arXiv preprint arXiv:2509.05773},
      year={2025},
}
```

