- [NPTEL2020 - Indian English Speech Dataset](#nptel2020---indian-english-speech-dataset)
  - [Crawl Information](#crawl-information)
  - [Dataset Quality](#dataset-quality)
  - [Suggestions and Future Works](#suggestions-and-future-works)
  - [Downloads](#downloads)
  - [Download via Torrent](#download-via-torrent)
  - [Crawl your own playlist](#crawl-your-own-playlist)
  - [Contact us](#contact-us)

# NPTEL2020 - Indian English Speech Dataset

A Speech-to-Text dataset scraped from [NPTEL](https://nptel.ac.in/course.html) for [Indo-English accent](https://en.wikipedia.org/wiki/Regional_differences_and_dialects_in_Indian_English), from Education Domain.

## Crawl Information

|Metadata|Details|
|-|-|
|Crawl Period|April 2020|
|Source|[NPTEL-HRD YouTube](https://www.youtube.com/user/nptelhrd) ([Playlist](https://www.youtube.com/playlist?list=UU640y4UvDAlya_WOj5U4pfA))|
|Crawl Type|Only videos with manually uploaded subtitles|
|Content License|[Creative Commons](https://www.youtube.com/t/creative_commons)|
|Language|English (most videos are in South-Asian accent)|
|Domain|General & Technical Education|
|Crawling and Processing Code|[Fast-KTSpeechCrawler](https://github.com/Prem-kumar27/Fast-KTSpeechCrawler)|
|Total Videos Crawled|19,500|
|Average Video Duration|40mins|
|Dataset Format|LibriSpeech (audio in `wav`, transcript in `txt`, metadata in `json`)|
|No. of chunks created|6,253,389 (6.2M)|
|Average chunk length|3 - 10 secs|
|Total no. of hours|15,700 hours|
|Total Compressed Dataset Size|1.1 TB (1.7 TB upon decompression)|

## Dataset Quality

The dataset was not manually annotated by us. We assume NPTEL has used Google ASR on top of which they have made reasonable amount of corrections.

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

To understand if the data we have crawled is useful, we sample 500k chunks from the train set and fine-tune an English ASR model using that for an epoch. We chose QuartzNet (Ultra-Tiny) pre-trained model for the fine-tuning because it was lightweight as well as very competitive in accuracy as seen above. (Training code can be found in this same repo)

On the Pure-Set, we observed an improvement in WER from 0.5034 (pre-trained model without LM) to 0.3207 (fine-tuned model without LM), which signified a promising scope to use the crawled dataset for much further improvement.

The pre-mature fine-tuned model can be found in the [GitHub Releases section](https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases) of the repo. (Currently due to lack of compute, we couldn't exhaustively use the data and find out results across different models & methods)

## Suggestions and Future Works

- Even though the dataset is noisy compared to publicly available datasets, we believe it would serve as a good intial data for building models.
- Especially this dataset focuses on South Asian English accent, and is of education domain.
- Even the raw audio from this dataset would be useful for pre-training ASR models like [Wav2Vec 2.0](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/).  
  ([As can be seen on this recent leaderboard](https://sites.google.com/view/englishasrchallenge/leaderboard#h.cteumzu5d5uo))
- For a better but closed dataset, check this recent competition: [IIT-M Speech Lab - Indian English ASR Challenge](https://sites.google.com/view/englishasrchallenge/home)  
  (could be available on request)

## Downloads

- [Sample Data (Pure-Set)](https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases/download/v0.1/nptel-pure-set.tar.gz)
- [Train, Test and Dev sets downloader scripts](/download_scripts)

## Download via Torrent

The `opus` [version](https://academictorrents.com/details/cc9dc56afd3055c7e0f021ec4f1824021558926c) of the dataset is hosted via academic torrents. The `opus` version is 10x smaller.

Please seed and make sure that your download ratio reaches `1.0`. Some torrent clients (e.g. `aria2c` have an issue being stuck at 99%).

## Crawl your own playlist

In some cases, we might need data containing a single speaker(TTS, Speaker recognition, etc). For that, choose a youtube playlist of your choice
and crawl it.

Please [check here](https://github.com/Prem-kumar27/Fast-KTSpeechCrawler#downloading-a-playlist) for the instructions to do that on the crawler we used.

## Contact us

For clarifications, please write on the GitHub Issues section.
