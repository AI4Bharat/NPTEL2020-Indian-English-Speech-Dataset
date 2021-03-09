# NPTEL2020 - Indian English Speech Dataset (Education domain)

## Crawl Information

|Metadata|Details|
|-|-|
|Crawl Period|April 2020|
|Source|[NPTEL-HRD YouTube](https://www.youtube.com/user/nptelhrd)|
|Crawl Type|Only videos with manually uploaded subtitles|
|Content License|[Creative Commons](https://www.youtube.com/t/creative_commons)
|Language|English (most videos are in South-Asian accent)|
|Domain|General & Technical Education|
|Crawling and Processing Code|[Fast-KTSpeechCrawler](https://github.com/Prem-kumar27/Fast-KTSpeechCrawler)|
|Total Videos Crawled|19,500|
|Average Video Duration|40mins|
|Dataset Format|LibriSpeech (audio in `wav`, transcript in `txt`)|
|No. of chunks created|6,253,389 (6.2M)|
|Average chunk length|3 - 10 secs|
|Total no. of hours|15,700 hours|
|Total Compressed Dataset Size|1.1 TB|

## Dataset Quality

The dataset was not manually annotated by us. We assume NPTEL use Google ASR on top of which they have made minor corrections.

We split the dataset as follows: (randomly sampled)

|Split|Number of chunks|
|-|-|
|Train set|5M|
|Validation set|625k|
|Test set|625k|
|Sample Set|1k|

Note:  
- Sample Set is a small subset manually annotated by us to compute the quality of data. We refer to it as **Pure Set**.
- We computed (in May 2020) the following results using the Pure Set for benchmarking purposes.
  - All the DL models listed below uses LM.
  - For Jasper and QuartzNet, we use models from [Nvidia's NeMo framework](https://github.com/NVIDIA/NeMo).
  - For DeepSpeech, we use Baidu's official model.
  - The benchmarking code for others can be [found here](https://github.com/narVidhai/Speech-Transcribers-Python).

|Model/Service|Word Error Rate|
|-------------|---------------|
|Actual transcripts|0.1451|
|Google Speech-to-Text|0.4895|
|AWS Transcribe|0.3438|
|Rev.ai (Temi Speech) API|0.3218|
|ESP-Net|0.4321|
|DeepSpeech v.0.7|0.5872|
|Nvidia Jasper|0.3939|
|QuartzNet (Ultra-Tiny)|0.3981|

To understand if the data we have crawled is useful, we sample 500k chunks from the train set and train it for an epoch. We chose QuartzNet (Ultra-Tiny) pre-trained model for the fine-tuning because it was lightweight as well as very competitive in accuracy.

On the Pure-Set, we observed an improvement in WER from 0.5034 (pre-trained model without LM) to 0.3207 (fine-tuned model without LM), which signified a promising scope to use the crawled dataset for much further improvement.

The pre-mature fine-tuned model can be found in the GitHub Releases section of the repo. (Currently due to lack of compute, we couldn't exhaustively use the data and propose the results)

## Suggestions and Future Works

- Even though the dataset is noisy compared to publicly available datasets, we believe it would serve as a good intial data for building models.
- Especially this dataset focuses on South Asian English accent, and is of education domain.
- Even the raw audio from this dataset would be useful for pre-training ASR models like [Wav2Vec 2.0](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)

## Downloads

--To Add--

## Contact us

For clarifications, please write on the GitHub Issues section.
