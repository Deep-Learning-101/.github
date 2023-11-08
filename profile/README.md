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

  <details close>

### ** 2020/03/08-2020/08/29 開發心得：**
投入約150天。通常我們是怎樣開始項目的研究與開發？首先會先盡可能的把3年內的學術論文或比賽等SOTA都查到，然後分工閱讀找到相關的數據集和論文及相關實作；同時會找到目前已有相關產品的公司(含新創)及他們提交的專利，這部份通常再花約30天的時間；通常就是透過 Google patens、paper with codes、arxiv等等。
聲紋識別這塊在對岸查到非常多的新創公司，例如: 國音智能在我們研究開發過程就是一直被當做目標的新創公司。可以先看一下上方的DEMO影片效果；然後介紹相關實驗結果前，避免之後有人還陸續踩到我們踩過的坑；需注意的是上述很多數據集都是放在對岸像是百度雲盤等，百度是直接封鎖台灣的IP，所以你打不開是很正常的；另外像是voxcelab是切成7份，下載完再合起來也要花上不少時間，aishell、CMDS, TIMIT 比起來相對好處理就是。
簡單總結為：1. 幾種 vector 的抽取 (i-vector, d-vector, x-vector) 跟 2. 模型架構 (CNN, ResNet) 和調參，再來就是 3. 評分方式 (LDA, PLDA (Probabilistic Linear Discriminant Analysis)) 等等幾種組合；我們也使用了 kaldi 其中內附的功能，光是 kaldi 就又投入了不少時間和精力 ! 其實比起自然語言處理做聲紋識別，最小的坑莫過於雖然數據集不是很容易獲取，但是聲音是可以自行用程式加工做切割合併，然後因為場景限制，錄聲紋時的時長頗短，還得處理非註冊聲紋的處理，所以前前後後花了很多時間在將相關的數據搭配評分模式調整，也算是個大工程。

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

### ** 2020/08/30-2021/01/25 開發心得：**
投入約150天。如同語音踩的坑來說，比較常碰到因為網路架構在做參數調整時導致loss壞掉等等，而因數據集造成的問題少很多，網路上也比較容易找到更多的數據集，然後也有非常多的比賽有各種模型架構的結果可以參考，但是一樣是英文數據，而語音坑最好的就是只要有了像是 aishell 等的數據集，你想要切割或合併成一個語音，都不是太大的問題；例如我們就是把數據集打散混合，再從中隨機挑選兩個人，然後再從中分別挑出語音做混合；如是長度不同，選擇短者為參考，將長者切到與短者相同；最後產出約 train： 5萬多筆，約 32小時、val：1萬多筆語音，約10小時、test：9,千多筆語音，約 6小時，而這個數據集是兩兩完全重疊，後來為了處理兩兩互不完全重疊，再次另外產出了這樣的數據集：train：9萬多筆語音，計112小時、val：2萬多筆語音，計 26.3 小時、test：2萬多筆語音，計 29.4 小時。
中間也意外發現了Google brain 的 wavesplit，在有噪音及兩個人同時講話情形下，感覺效果還不差，但沒找到相關的code，未能進一步驗證或是嘗試更改數據集。還有又是那位有一起用餐之緣的深度學習大神 Yann LeCun繼發文介紹 完去噪後，又發文介紹了語音分離；後來還有像是最早應用在NLP的Transformer等Dual-path RNN (DP-RNN) 或 DPT-NET (Dual-path transformer) 等應用在語音增強/分割，另外VoiceFilter、TasNet 跟 Conv-TasNet還有sudo-rm等等也是語音分割相關，當然更不能錯過臺大電機李宏毅老師一篇SSL-pretraining-separation的論文 (務必看完臺大電機李宏毅老師的影片)，最後也是多虧李老師及第一作者黃同學的解惑，然後小夥伴們才又更深入的確認並且解決問題。
這裡做數據時相對簡單一點，直接打散混合，再從中隨機挑選兩個人，然後分別挑出語音做混合，若長度不同，選擇短者為參考，將長者切到與短者相同，兩兩完全重疊或者兩兩互不完全重疊等都對效果有不小的影響；同時也研究了Data Parallel 跟 Distributed Data Parallel 的差異，但是如何才能在 CPU 上跑得又快又準才是落地的關鍵

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

### ** 2018/10/15-2019/02/10 開發心得：**
投入約120天，早期想上線需要不少計算資源 (沒有昂貴的GPU想上線簡直是難如登天，好險時代在進步，現在CPU就能搞定)。記得我2018從老闆口中第一次聽到新項目是機器閱讀理解時，一頭霧水不知道是在幹麼，Google後突然發現這還真是它X的超級難的東西，而當時落地場景是要解決機器人在博物館或者展場的Q&A，不想再預先建一堆關鍵字與正規表示式來幫相似度和分類做前處理。

但機器閱讀理解坑真的不小，首先當然是數據，公開數據有SQuAD 1.0和2.0，但這是英文，你想用在中文 ? 你可以自己試試啦，再來有了個中文的CMRC，但用得是對岸用語跟簡體中文，而且數據格式不太一樣；後來台達電放出了DRCD還有科技部辦的科技大擂台，依然有格式不同的問題，數據量真的不太夠，所以想要落地你真的得要自己標註。

為了解決像是多文章還有問非文章內問題，還有公開數據要嘛英文不然就是簡體中文或對岸用語，然後本地化用語的數據實在不足的狀況，小夥伴們真的很給力，我們也用機器翻譯SQuAD 1.0和2.0還有自己手工爬維基百科跟開發了數據標註系統自己標註 ! 不得不說小夥伴們真的是投入超多精神在機器閱讀理解，更在Deep Learning 101做了分享。

  <summary>Machine Reading Comprehension (機器閱讀理解)</summary>

  - [中文機器閱讀理解](https://www.twman.org/AI/NLP/MRC)
    - [機器閱讀理解綜述(一)](https://zhuanlan.zhihu.com/p/80905984)
    - [機器閱讀理解綜述(二)](https://zhuanlan.zhihu.com/p/80980403)
    - [機器閱讀理解綜述(三)](https://zhuanlan.zhihu.com/p/81126870)
    - [機器閱讀理解探索與實踐](https://zhuanlan.zhihu.com/p/109309164)
    - [什麼是機器閱讀理解？跟自然語言處理有什麼關係？](https://communeit.medium.com/%E4%BB%80%E9%BA%BC%E6%98%AF%E6%A9%9F%E5%99%A8%E9%96%B1%E8%AE%80%E7%90%86%E8%A7%A3-%E8%B7%9F%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E6%9C%89%E4%BB%80%E9%BA%BC%E9%97%9C%E4%BF%82-b02fb6ccb6e9)

  </details>

  <details close>

### ** 2019/12/02-2020/02/29 開發心得：**
記得前後兩次陸續投入總計約100天。或許有人會發現為何在分享這幾篇自然語言會強調中文數據？最好理解的說法就是中文是基於字表示再加上中文斷詞的效果，比起每個單詞只需空格來表示的英文硬是麻煩點。命名實體識別 (Named Entity Recognition, NER) 是指將語句中的元素分成預先定義的類別 (開放域來說包括實體、時間和數字3個大類，人名、地名、組織名、機構名、時間、日期、數量和名字等7個小類，特定領域就像是藥名、疾病等類別)。要應用在那方面？像是關係抽取、對話意圖理解、輿情分析、對話NLU任務等等都用得上，更廣義的就屬填槽 (Slot-Filling) 了。

最早 (2019/08時) 我們需處理的場景是針對電話助理的對話內容 (就是APP幫你接電話跟對方對話) 在語音識別後跟語音合成前的處理，印像中沒做到非常深入；後來剛好招聘到熟悉NER這部份的小夥伴們，剛好一直想把聊天對話做個流程處理 (多輪對話的概念) ，就再花了點時間當做上手，因為不想依賴大量關鍵字和正規表示式做前處理，中間試了不少數據集，還做了像是用拼音、注音等，或者品牌定義等超多的實驗，甚至還一度想硬整合 RASA 等等的開源套件，也嘗試用了 "改寫" 來修正對話內容，去識別出語句中的重點字。至於這個的數據標據就真的更累人，意外找到一個蠻好用的標註系統 ChineseAnnotator，然後我們就瘋狂開始標註 !

  <summary>Named Entity Recognition (命名實體識別)</summary>

  - [中文命名實體識別](https://www.twman.org/AI/NLP/NER)

  </details>

  <details close>

### ** 2019/11/20-2020/02/29 開發心得：**
投入約100天，早期上線成本資源頗高，現在就沒這問題；這個項目堪稱是在NLP這個坑裡投入第二多的，記得當時的場景是機器人在商場裡回答問題所顯示出來的文字會有一些ASR的錯字，但是問題一樣卡在數據集，還因此讓小夥伴們花了好長時間辛苦去標註 XD，但看看現在效果，我想這是值得的 ! 記得一開始是先依賴 pycorrector，然後再換 ConvSeq2Seq，當然 bert 也做了相關優化實驗，中間一度被那三番二次很愛嗆我多讀書，從RD轉職覺得自己很懂做產品的PM拿跟百度對幹，從一開始的看實驗結果輸，到後來贏了，卻又自己亂測說還是不夠好之類的叭啦叭啦，說實話，你最後不也人設垮了然後閃人 ~ 攤手 ~ 

現在看看這截圖效果，不是蠻勵害的嗎 ? 真的想說這社會真的充滿一堆人設嚇死人的人，無敵愛嘴砲 ! 搞的為了滿足那位人設比天高的需求，真的是想了像是用拼音還兼NER來整合的好幾種方法 ! 那文本糾錯會有什麼坑呢？：數據啊、格式啊 !!! 還有幾個套件所要處理的目標不太一樣，有的可以處理疊字有的可以處理連錯三個字，還有最麻煩的就是斷字了，因為現有公開大家最愛用的仍舊是Jieba，即便它是有繁中版，當然也能試試 pkuseg，但就是差了點感覺。

  <summary>Correction (糾錯)</summary>

  - [中文文本糾錯](https://www.twman.org/AI/NLP/Correction)

  </details>

  <details close>

### ** 2019/11/10-2019/12/10 開發心得：**
最早我們是透過 Hierarchical Attention Networks for Document Classification (HAN) 的實作，來修正並且以自有數據進行訓練；但是這都需要使用到騰訊放出來的近16 GB 的 embedding：Tencent_AILab_ChineseEmbedding_20190926.txt，如果做推論，這會是個非常龐大需載入的檔案，直到後來 Huggingface 橫空出世，解決了 bert 剛出來時，很難將其當做推論時做 embedding 的 service (最早出現的是 bert-as-service)；同時再接上 BiLSTM 跟 Attention。CPU (Macbook pro)：平均速度：約 0.1 sec/sample，總記憶體消耗：約 954 MB (以 BiLSTM + Attention 為使用模型)。
引用 Huggingface transformers 套件 bert-base-chinese 模型作為模型 word2vec (embedding) 取代騰訊 pre-trained embedding
優點：API 上線時無須保留龐大的 Embedding 辭典,避免消耗大量記憶體空間，但BERT 相較於傳統辭典法能更有效處理同詞異義情況，更簡單且明確的使用 BERT 或其他 Transformers-based 模型
缺點：Embedding後的結果不可控制，BERT Embedding 維度較大,在某些情況下可能造成麻煩

  <summary>Classification (分類)</summary>

  - [中文文本分類](https://www.twman.org/AI/NLP/Classification)

  </details>

  <details close>

### ** 2019/10/15-2019/11/30 開發心得：**
投入約45天，那時剛好遇到 albert，但最後還是被蒸溜給幹掉；會做文本相似度主要是要解決當機器人收到ASR識別後的問句，在進到關鍵字或正規表示式甚至閱讀理解前，藉由80/20從已存在的Q&A比對，然後直接解答；簡單來說就是直接比對兩個文句是否雷同，這需要準備一些經典/常見的問題以及其對應的答案，如果有問題和經典/常見問題很相似，需要可以回答其經典/常見問題的答案；畢竟中文博大精深，想要認真探討其實非常難，像是廁所在那裡跟洗手間在那，兩句話的意思真的一樣，但字卻完全不同；至於像是我不喜歡你跟你是個好人，這就是另一種相似度了 ~ xDDD ! 那關於訓練數據資料，需要將相類似的做為集合，這部份就需要依賴文本分類；你可能也聽過 TF-IDF 或者 n-gram 等，這邊就不多加解釋，建議也多查查，現在 github 上可以找到非常的範例程式碼，建議一定要先自己動手試試看 !

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