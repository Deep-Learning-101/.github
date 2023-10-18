# <p align="center">Deep Learning 101</a></p>

## <p align="center">The top private AI Meetup in Taiwan, launched on 2016</a></p>

<p align="center">
<img src="./images/DeepLearning101.JPG" width="50%" />

<p align="center">
<a href="https://www.facebook.com/groups/525579498272187/">台灣人工智慧社團</a>
</p>

<p align="center">
http://DeepLearning101.TWMAN.ORG
<p align="center">
https://huggingface.co/DeepLearning101  
<p align="center">
https://www.youtube.com/@DeepLearning101

##

### [Speech Processing( 語音處理)](https://github.com/Deep-Learning-101/Speech-Processing-Paper)：**[那些語音處理踩的坑](https://blog.twman.org/2021/04/ASR.html)**：[針對訪談或對話進行分析與識別](https://www.twman.org/AI/ASR)。

<details open>
<summary><strong>語音處理</strong></summary>
  
  <details open>
  <summary>Speech Recognition (語音識別)</summary>

  - [中文語音識別](https://www.twman.org/AI/ASR)
    - [語音識別質檢+時間戳：Whisper Large V2](https://huggingface.co/spaces/DeepLearning101/Speech-Quality-Inspection_whisperX)
  - [Whisper](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/Whisper.md)
  - [WeNet](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/WeNet.md)
  - [FunASR](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/FunASR.md)

  </details>

  <details open>
  <summary>Speaker Recognition (聲紋識別)</summary>

  - [中文語者(聲紋)識別](https://www.twman.org/AI/ASR/SpeakerRecognition)
  - [WeSpeaker](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/WeSpeaker.md)
  - [SincNet](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/SincNet.md)

  </details>

  <details close>
  <summary>Speech Enhancement (語音增強)</summary>

  - [中文語音增強(去噪)](https://www.twman.org/AI/ASR/SpeechEnhancement)
    - [語音質檢+噪音去除：Meta Denoiser](https://huggingface.co/spaces/DeepLearning101/Speech-Quality-Inspection_Meta-Denoiser)
  - [Denoiser](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/Denoiser.md)

  </details>

  <details close>
  <summary>Speech Separation (語音分離)</summary>

  - [中文語者分離(分割)](https://www.twman.org/AI/ASR/SpeechSeparation)
  - [Mossformer](https://github.com/Deep-Learning-101/Speech-Processing-Paper/blob/main/Mossformer.md)
  - [TOLD@FASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/egs/callhome/TOLD)
    - [TOLD能對混疊語音建模的說話人日誌框架](https://zhuanlan.zhihu.com/p/650346578)

  </details>

  <details close>
  <summary>Speech Synthesis (語音合成)</summary>

  - [Rectified Flow Matching 語音合成，上海交大開源](https://www.speechhome.com/blogs/news/1712396018944970752)：https://github.com/cantabile-kwok/VoiceFlow-TTS
  - [新一代開源語音庫CoQui TTS衝到了GitHub 20.5k Star](https://zhuanlan.zhihu.com/p/661291996)：https://github.com/coqui-ai/TTS/
  - [清華大學LightGrad-TTS，且流式實現](https://zhuanlan.zhihu.com/p/656012430)：https://github.com/thuhcsi/LightGrad
  - [出門問問MeetVoice, 讓合成聲音以假亂真](https://zhuanlan.zhihu.com/p/92903377)
  - [VALL-E：微軟全新語音合成模型可以在3秒內復制任何人的聲音](https://zhuanlan.zhihu.com/p/598473227)
  - [BLSTM-RNN、Deep Voice、Tacotron…你都掌握了吗？一文总结语音合成必备经典模型（一）](https://new.qq.com/rain/a/20221204A02GIT00)
  - [Tacotron2、GST、Glow-TTS、Flow-TTS…你都掌握了吗？一文总结语音合成必备经典模型（二）](https://cloud.tencent.com/developer/article/2250062)
  - Bark：https://github.com/suno-ai/bark
      - [最強文本轉語音工具：Bark，本地安裝+雲端部署+在線體驗詳細教程](https://zhuanlan.zhihu.com/p/630900585)
      - [使用Transformers 優化文本轉語音模型Bark](https://zhuanlan.zhihu.com/p/651951136)

  </details>
</details>

##

### [Natural Language Processing, NLP (自然語言處理)](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper)：**[那些自然語言處理踩的坑](https://blog.twman.org/2021/04/NLP.html)**：[針對文檔進行分析與擷取](https://www.twman.org/AI/NLP)。

#### [大型語言模型(Large Language Model，LLM)，想要嗎？](https://blog.twman.org/2023/04/GPT.html)
#### [基於機器閱讀理解的指令微調的統一信息抽取框架之診斷書醫囑擷取分析](https://blog.twman.org/2023/07/HugIE.html)：https://huggingface.co/spaces/DeepLearning101/IE101TW

<details open>
<summary><strong>自然語言處理</strong></summary>

  <details open>
  <summary>Large Language Model (大語言模型)</summary>

  - [LangChain](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper#langchain)
  - [Retrieval Augmented Generation](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper#rag)
  - [LLM Model](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper#llm-%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%B4%B9)

  </details>
  
  <details open>
  <summary>Information/Event Extraction (資訊/事件擷取)</summary>

  - [HugNLP](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/HugNLP.md)
  - [DeepKE](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/DeepKE.md)
  - [ERINE-Layout](https://github.com/Deep-Learning-101/Natural-Language-Processing-Paper/blob/main/ERNIE-Layout.md)
  - [UIE @ PaddleNLP](https://huggingface.co/spaces/DeepLearning101/PaddleNLP-UIE)
    - https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie


  </details>

  <details close>
  <summary>Machine Reading Comprehension (機器閱讀理解)</summary>

  - [中文機器閱讀理解](https://www.twman.org/AI/NLP/MRC)
    - [繁體中文閱讀理解：Bert](https://huggingface.co/spaces/DeepLearning101/Reading-Comprehension_Bert)

  </details>

  <details close>
  <summary>Named Entity Recognition (命名實體識別)</summary>

  - [中文命名實體識別](https://www.twman.org/AI/NLP/NER)

  </details>

  <details close>
  <summary>Correction (糾錯)</summary>

  - [中文文本糾錯](https://www.twman.org/AI/NLP/Correction)

  </details>

  <details close>
  <summary>Classification (分類)</summary>

  - [中文文本分類](https://www.twman.org/AI/NLP/Classification)

  </details>

  <details close>
  <summary>Similarity (相似度)</summary>

  - [中文文本相似度](https://www.twman.org/AI/NLP/Similarity)

  </details>

</details>




##

### [Computer vision (電腦視覺)](https://www.twman.org/AI/CV)：[針對物件或場景影像進行分析與偵測](https://github.com/Deep-Learning-101/Computer-Vision-Paper)。

#### [用PaddleOCR的PPOCRLabel來微調醫療診斷書和收據](https://blog.twman.org/2023/07/wsl.html)


<details open>
<summary><strong>圖像處理：</strong></summary>

  <details close>
  <summary>Optical Character Recognition (光學字元辨識)</summary>

  - [繁體中文醫療診斷書和收據OCR：PaddleOCR](https://huggingface.co/spaces/DeepLearning101/OCR101TW)
  - PaddleOCR

  </details>

  <details open>
  <summary>Document Layout Analysis (文件結構分析)</summary>

  - [arXiv-2020_LayoutLM](https://github.com/Deep-Learning-101/Computer-Vision-Paper/blob/main/LayoutLM.md)
  - [arXiv-2021_LayoutLMv2](https://github.com/Deep-Learning-101/Computer-Vision-Paper/blob/main/LayoutLMv2.md)
  - arXiv-2021_LayoutXLM
  - arXiv-2022_LayoutLMv3
    
  </details>

  <details close>
  <summary>Document Understanding (文件理解)</summary>    
  </details>

  <details close>
  <summary>Object Detection (物件偵測)</summary>
  </details>

  <details close>
  <summary>Handwriting Recognition (手寫識別)</summary>
  </details>

  <details close>
  <summary>Face Recognition (人臉識別)</summary>   
  </details>

</details>