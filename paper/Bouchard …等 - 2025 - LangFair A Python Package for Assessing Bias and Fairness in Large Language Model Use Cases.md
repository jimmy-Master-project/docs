# Bouchard …等 - 2025 - LangFair A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases

---

<!-- Page 1 -->

LangFair

LangFair: A Python Package for Assessing Bias and Fairness in Large Language Model Use Cases

arXiv:2501.03112v1 [cs.CL] 6 Jan 2025

Dylan Bouchard1 Mohit Singh Chauhan1 David Skarbrevik1 Viren Bajaj1 Zeya Ahmad1

dylan.bouchard@cvshealth.com mohitsingh.chauhan@cvshealth.com david.skarbrevik@cvshealth.com bajajv@aetna.com zeya.ahmad@cvshealth.com The authors are the current maintainers of LangFair, and additionally have the following affiliations: 1 CVS Health Corporation

Abstract Large Language Models (LLMs) have been observed to exhibit bias in numerous ways, potentially creating or worsening outcomes for specific groups identified by protected attributes such as sex, race, sexual orientation, or age. To help address this gap, we introduce langfair, an open-source Python package that aims to equip LLM practitioners with the tools to evaluate bias and fairness risks relevant to their specific use cases. The package offers functionality to easily generate evaluation datasets, comprised of LLM responses to use-case-specific prompts, and subsequently calculate applicable metrics for the practitioner’s use case. To guide in metric selection, LangFair offers an actionable decision framework. Keywords: large language models, algorithmic fairness, artificial intelligence, machine learning, Python

1. Introduction Traditional machine learning (ML) fairness toolkits like AIF360 (Bellamy et al., 2018), Fairlearn (Weerts et al., 2023), Aequitas (Saleiro et al., 2018) and others (Vasudevan and Kenthapadi, 2020; Wexler et al., 2019; Tensorflow, 2020) have laid crucial groundwork. These toolkits offer various metrics and algorithms that focus on assessing and mitigating bias and fairness through different stages of the ML lifecycle. While the fairness assessments offered by these toolkits include a wide variety of generic fairness metrics, which can also apply to certain LLM use cases, they are not tailored to the generative and context-dependent nature of LLMs.1 LLMs are used in systems that solve tasks such as recommendation, classification, text generation, and summarization. In practice, these systems try to restrict the responses of the LLM to the task at hand, often by including task-specific instructions in system or user prompts. When the LLM is evaluated without taking the set of task-specific prompts into account, the evaluation metrics are not representative of the system’s true performance. Representing the system’s actual performance is especially important when evaluating its 1. The toolkits mentioned here offer fairness metrics for classification. In a similar vein, the recommendation fairness metrics offered in FaiRLLM (Zhang et al., 2023) can be applied to ML recommendation systems as well as LLM recommendation use cases.

1

---

<!-- Page 2 -->

Bouchard et al.

outputs for bias and fairness risks because they pose real harm to the user and, by way of repercussions, the system developer. Most evaluation tools, including those that assess bias and fairness risk, evaluate LLMs at the model-level by calculating metrics based on the responses of the LLMs to static benchmark datasets of prompts (Rudinger et al., 2018; Zhao et al., 2018; Webster et al., 2018; Levy et al., 2021; Nadeem et al., 2020; Bartl et al., 2020; Nangia et al., 2020; Felkner et al., 2024; Barikeri et al., 2021; Kiritchenko and Mohammad, 2018; Qian et al., 2022; Gehman et al., 2020; Dhamala et al., 2021; Huang et al., 2023; Nozza et al., 2021; Parrish et al., 2022; Li et al., 2020; Krieg et al., 2023) that do not consider prompt-specific risks and are often independent of the task at hand. Holistic Evaluation of Language Models (HELM) (Liang et al., 2023), DecodingTrust (Wang et al., 2023), and several other toolkits (Srivastava et al., 2022; Huang et al., 2024; Gao et al., 2024; Nazir et al., 2024; Huggingface, 2022) follow this paradigm. In this paper, we introduce langfair, an open-source Python package for conducting bias and fairness assessments of LLM use cases.2 LangFair complements the aforementioned frameworks because it follows a bring your own prompts (BYOP) approach, which allows users to tailor the bias and fairness evaluation to their use case by computing metrics using LLM responses to user-provided prompts. This addresses the need for a task-based bias and fairness evaluation tool that accounts for prompt-specific risk for LLMs.3 To guide the selection of bias and fairness metrics, LangFair offers an actionable decision framework, discussed in detail in the project’s companion paper, Bouchard (2024). Furthermore, LangFair is designed for real-world LLM-based systems that require governance audits. LangFair focuses on calculating metrics from LLM responses only, which is more practical for real-world testing where access to internal states of model to retrieve embeddings or token probabilities is difficult. An added benefit is that output-based metrics, which are focused on the downstream task, have shown to be potentially more reliable than metrics derived from embeddings or token probabilities (Goldfarb-Tarrant et al., 2021; Delobelle et al., 2022).

2. Generation of Evaluation Datasets The langfair.generator module offers two classes, ResponseGenerator and CounterfactualGenerator, which aim to enable user-friendly construction of evaluation datasets for text generation use cases. ResponseGenerator class.

To streamline generation of evaluation datasets, the Respon-

seGenerator class wraps an instance of a langchain LLM and leverages asynchronous

generation with asyncio. To implement, users simply pass a list of prompts (strings) to the ResponseGenerator.generate responses method, which returns a dictionary containing prompts, responses, and applicable metadata. CounterfactualGenerator class.

In the context of LLMs, counterfactual fairness can be assessed by constructing counterfactual input pairs (Gallegos et al., 2024; Bouchard, 2024), 2. The repository for langfair can be found at https://github.com/cvs-health/langfair. 3. Experiments in Wang et al. (2023) demonstrate that prompt content has substantial influence on the likelihood of biased LLM responses.

2

---

<!-- Page 3 -->

LangFair

comprised of prompt pairs that mention different protected attribute groups but are otherwise identical, and measuring the differences in the corresponding generated output pairs. These assessments are applicable to use cases that do not satisfy fairness through unawareness (FTU), meaning prompts contain mentions of protected attribute groups. To address this, the CounterfactualGenerator class offers functionality to check for FTU, construct counterfactual input pairs, and generate corresponding pairs of responses asynchronously using a langchain LLM instance.4 Off the shelf, the FTU check and creation of counterfactual input pairs can be done for gender and race/ethnicity, but users may also provide a custom mapping of protected attribute words to enable this functionality for other attributes as well.

3. Bias and Fairness Evaluations for Focused Use Cases Following Bouchard (2024), evaluation metrics are categorized according to the risks they assess (toxicity, stereotypes, counterfactual unfairness, and allocational harms), as well as the use case task (text generation, classification, and recommendation).5 Table 1 maps the classes contained in the langfair.metrics module to these risks. These classes are discussed in detail below. Table 1: Classes for Computing Evaluation Metrics in langfair.metrics Class Risk Assessed Applicable Tasks ToxicityMetrics StereotypeMetrics CounterfactualMetrics RecommendationMetrics ClassificationMetrics

Toxicity Stereotypes Counterfactual fairness Counterfactual fairness Allocational harms

Text generation Text generation Text generation Recommendation Classification

Toxicity Metrics The ToxicityMetrics class facilitates simple computation of toxicity metrics from a user-provided list of LLM responses. These metrics leverage a pre-trained toxicity classifier that maps a text input to a toxicity score ranging from 0 to 1 (Gehman et al., 2020; Liang et al., 2023). For off-the-shelf toxicity classifiers, the ToxicityMetrics class provides four options: two classifiers from the detoxify package, roberta-hate-speechdynabench-r4-target from the evaluate package, and toxigen available on HuggingFace.6 For additional flexibility, users can specify an ensemble of the off-the-shelf classifiers offered or provide a custom toxicity classifier object. Stereotype Metrics To measure stereotypes in LLM responses, the StereotypeMetrics class offers two categories of metrics: metrics based on word cooccurrences and metrics that leverage a pre-trained stereotype classifier. Metrics based on word cooccurrences aim to assess relative cooccurrence of stereotypical words with certain protected attribute words. 4. In practice, a FTU check consists of parsing use case prompts for mentions of protected attribute groups. 5. Note that text generation encompasses all use cases for which output is text, but does not belong to a predefined set of elements (as with classification and recommendation). 6. https://github.com/unitaryai/detoxify; https://github.com/huggingface/evaluate; https://github. com/microsoft/TOXIGEN

3

---

<!-- Page 4 -->

Bouchard et al.

On the other hand, stereotype-classifier-based metrics leverage the wu981526092/SentenceLevel-Stereotype-Detector classifier available on HuggingFace (Zekun et al., 2023) and compute analogs of the aforementioned toxicity-classifier-based metrics (Bouchard, 2024).7 Counterfactual Fairness Metrics for Text Generation The CounterfactualMetrics class offers two groups of metrics to assess counterfactual fairness in text generation use cases. The first group of metrics leverage a pre-trained sentiment classifier to measure sentiment disparities in counterfactually generated outputs (see Huang et al. (2020) for further details). This class uses the vaderSentiment classifier by default but also gives users the option to provide a custom sentiment classifier object.8 The second group of metrics addresses a stricter desiderata and measures overall similarity in counterfactually generated outputs using well-established text similarity metrics (Bouchard, 2024). Counterfactual Fairness Metrics for Recommendation The RecommendationMetrics class is designed to assess counterfactual fairness for recommendation use cases. Specifically, these metrics measure similarity in generated lists of recommendations from counterfactual input pairs. Metrics may be computed pairwise (Bouchard, 2024), or attribute-wise (Zhang et al., 2023). Fairness Metrics for Classification When LLMs are used to solve classification problems, traditional machine learning fairness metrics may be applied, provided that inputs can be mapped to a protected attribute. To this end, the ClassificationMetrics class offers a suite of metrics to address unfair classification by measuring disparities in predicted prevalence, false negatives, or false positives. When computing metrics using the ClassificationMetrics class, the user may specify whether to compute these metrics as pairwise differences (Bellamy et al., 2018) or pairwise ratios (Saleiro et al., 2018).

4. Semi-Automated Evaluation AutoEval class.

To streamline assessments for text generation use cases, the AutoEval class conducts a multi-step process (each step is described in detail above) for a comprehensive fairness assessment. Specifically, these steps include metric selection (based on whether FTU is satsified), evaluation dataset generation from user-provided prompts with a user-provided LLM, and computation of applicable fairness metrics. To implement, the user is required to supply a list of prompts and an instance of langchain LLM. Below we provide a basic example demonstrating the execution of AutoEval.evaluate with a gemini-pro instance.9

from langchain_google_vertexai import ChatVertexAI from langfair.auto import AutoEval llm = ChatVertexAI(model_name=’gemini-pro’) auto_object = AutoEval(prompts=prompts, langchain_llm=llm) results = await auto_object.evaluate()

Listing 1: AutoEval Example 7. https://huggingface.co/wu981526092/Sentence-Level-Stereotype-Detector 8. https://github.com/cjhutto/vaderSentiment 9. Note that this example assumes the user has already set up their VertexAI credentials and sampled a list of prompts from their use case prompts.

4

---

<!-- Page 5 -->

LangFair

Under the hood, the AutoEval.evaluate method 1) checks for FTU, 2) generates responses and counterfactual responses (if FTU is not satisfied), and 3) calculates applicable metrics for the use case.10 This process flow is depicted in Figure 1.

Figure 1: Flowchart of internal design of AutoEval.evaluate method.

5. Conclusions In this paper, we introduced langfair, an open-source Python package that aims to equip LLM practitioners with the tools to evaluate bias and fairness risks relevant to their specific use cases. The package offers functionality to easily generate evaluation datasets, comprised of LLM responses to use-case-specific prompts, and subsequently calculate applicable metrics for the practitioner’s use case. To aid practitioners, we provide numerous supporting resources, including API documentation, tutorial notebooks, and a technical companion paper (Bouchard, 2024).11

Author Contributions Dylan Bouchard was the principal developer and researcher of the LangFair project, responsible for conceptualization, methodology, and software development of the langfair library. Mohit Singh Chauhan was the architect behind the structural design of the langfair library and helped lead the software development efforts. David Skarbrevik was the primary author of LangFair’s documentation, helped implement software engineering best practices, and contributed to software development. Viren Bajaj wrote unit tests, contributed to the software development, and helped implement software engineering best practices. Zeya Ahmad contributed to the software development. 10. The ‘AutoEval‘ class is designed specifically for text generation use cases. Applicable metrics include toxicity metrics, stereotype metrics, and, if FTU is not satisfied, counterfactual fairness metrics. 11. https://cvs-health.github.io/langfair/latest/index.html

5

---

<!-- Page 6 -->

Bouchard et al.

Acknowledgments We wish to thank Piero Ferrante, Blake Aber, Xue (Crystal) Gu, and Zirui Xu for their helpful suggestions.

References Soumya Barikeri, Anne Lauscher, Ivan Vulić, and Goran Glavaš. Redditbias: A real-world resource for bias evaluation and debiasing of conversational language models, 2021. URL https://arxiv.org/abs/2106.03521. Marion Bartl, Malvina Nissim, and Albert Gatt. Unmasking contextual stereotypes: Measuring and mitigating bert’s gender bias. In Marta R. Costa-jussà, Christian Hardmeier, Kellie Webster, and Will Radford, editors, Proceedings of the Second Workshop on Gender Bias in Natural Language Processing, 2020. Rachel K. E. Bellamy, Kuntal Dey, Michael Hind, Samuel C. Hoffman, Stephanie Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta, Aleksandra Mojsilovic, Seema Nagar, Karthikeyan Natesan Ramamurthy, John Richards, Diptikalyan Saha, Prasanna Sattigeri, Moninder Singh, Kush R. Varshney, and Yunfeng Zhang. AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias, October 2018. URL https://arxiv.org/abs/1810.01943. Dylan Bouchard. An actionable framework for assessing bias and fairness in large language model use cases, 2024. URL https://arxiv.org/abs/2407.10853. Pieter Delobelle, Ewoenam Tokpo, Toon Calders, and Bettina Berendt. Measuring fairness with biased rulers: A comparative study on bias metrics for pre-trained language models. In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1693–1706, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/ 2022.naacl-main.122. URL https://aclanthology.org/2022.naacl-main.122. Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, KaiWei Chang, and Rahul Gupta. Bold: Dataset and metrics for measuring biases in openended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’21, page 862–872, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383097. doi: 10.1145/3442188. 3445924. URL https://doi.org/10.1145/3442188.3445924. Virginia K. Felkner, Ho-Chun Herbert Chang, Eugene Jang, and Jonathan May. Winoqueer: A community-in-the-loop benchmark for anti-lgbtq+ bias in large language models, 2024. URL https://arxiv.org/abs/2306.15087. Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. Bias and fairness in large language models: A survey, 2024. URL https://arxiv.org/abs/2309.00770. 6

---

<!-- Page 7 -->

LangFair

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/12608602. Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. In Findings, 2020. URL https://api.semanticscholar.org/CorpusID:221878771. Seraphina Goldfarb-Tarrant, Rebecca Marchant, Ricardo Muñoz Sanchez, Mugdha Pandya, and Adam Lopez. Intrinsic bias metrics do not correlate with application bias, 2021. URL https://arxiv.org/abs/2012.15859. Po-Sen Huang, Huan Zhang, Ray Jiang, Robert Stanforth, Johannes Welbl, Jack Rae, Vishal Maini, Dani Yogatama, and Pushmeet Kohli. Reducing sentiment bias in language models via counterfactual evaluation, 2020. URL https://arxiv.org/abs/1911.03064. Yue Huang, Qihui Zhang, Philip S. Y, and Lichao Sun. Trustgpt: A benchmark for trustworthy and responsible large language models, 2023. URL https://arxiv.org/abs/2306. 11507. Yue Huang, Lichao Sun, Haoran Wang, Siyuan Wu, Qihui Zhang, Yuan Li, Chujie Gao, Yixin Huang, Wenhan Lyu, Yixuan Zhang, Xiner Li, Hanchi Sun, Zhengliang Liu, Yixin Liu, Yijue Wang, Zhikun Zhang, Bertie Vidgen, Bhavya Kailkhura, Caiming Xiong, Chaowei Xiao, Chunyuan Li, Eric P. Xing, Furong Huang, Hao Liu, Heng Ji, Hongyi Wang, Huan Zhang, Huaxiu Yao, Manolis Kellis, Marinka Zitnik, Meng Jiang, Mohit Bansal, James Zou, Jian Pei, Jian Liu, Jianfeng Gao, Jiawei Han, Jieyu Zhao, Jiliang Tang, Jindong Wang, Joaquin Vanschoren, John Mitchell, Kai Shu, Kaidi Xu, Kai-Wei Chang, Lifang He, Lifu Huang, Michael Backes, Neil Zhenqiang Gong, Philip S. Yu, Pin-Yu Chen, Quanquan Gu, Ran Xu, Rex Ying, Shuiwang Ji, Suman Jana, Tianlong Chen, Tianming Liu, Tianyi Zhou, William Yang Wang, Xiang Li, Xiangliang Zhang, Xiao Wang, Xing Xie, Xun Chen, Xuyu Wang, Yan Liu, Yanfang Ye, Yinzhi Cao, Yong Chen, and Yue Zhao. Trustllm: Trustworthiness in large language models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=bWUU0LwwMp. Huggingface. Github - huggingface/evaluate: Evaluate: A library for easily evaluating machine learning models and datasets., 2022. URL https://github.com/huggingface/ evaluate. Svetlana Kiritchenko and Saif M. Mohammad. Examining gender and race bias in two hundred sentiment analysis systems, 2018. URL https://arxiv.org/abs/1805.04508. Klara Krieg, Emilia Parada-Cabaleiro, Gertraud Medicus, Oleg Lesota, Markus Schedl, and Navid Rekabsaz. Grep-biasir: A dataset for investigating gender representation bias in information retrieval results. In Proceedings of the 2023 Conference on Human 7

---

<!-- Page 8 -->

Bouchard et al.

Information Interaction and Retrieval, CHIIR ’23, page 444–448, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400700354. doi: 10.1145/3576840. 3578295. URL https://doi.org/10.1145/3576840.3578295. Shahar Levy, Koren Lazar, and Gabriel Stanovsky. Collecting a large-scale gender bias dataset for coreference resolution and machine translation, 2021. Tao Li, Daniel Khashabi, Tushar Khot, Ashish Sabharwal, and Vivek Srikumar. UNQOVERing stereotyping biases via underspecified questions. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3475–3489, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.311. URL https://aclanthology.org/ 2020.findings-emnlp.311. Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models, 2023. URL https://arxiv.org/abs/2211.09110. Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models, 2020. Nikita Nangia, Clara Vania, Rasika Bhalerao, and Samuel R. Bowman. CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, Online, November 2020. Association for Computational Linguistics. Arshaan Nazir, Thadaka Kalyan Chakravarthy, David Amore Cecchini, Thadaka Kalyan Chakravarthy, Rakshit Khajuria, Prikshit Sharma, Ali Tarik Mirik, Veysel Kocaman, and David Talby. LangTest: A comprehensive evaluation library for custom LLM and NLP models. Software Impacts, 19(100619), 2024. doi: 10.1016/j.simpa.2024.100619. Debora Nozza, Federico Bianchi, and Dirk Hovy. ”HONEST: Measuring hurtful sentence completion in language models”. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2398–2406, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.191. URL https://aclanthology.org/2021. naacl-main.191. Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. BBQ: A hand-built bias benchmark 8

---

<!-- Page 9 -->

LangFair

for question answering. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 2086–2105, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.165. URL https://aclanthology.org/2022.findings-acl. 165. Rebecca Qian, Candace Ross, Jude Fernandes, Eric Smith, Douwe Kiela, and Adina Williams. Perturbation augmentation for fairer nlp, 2022. URL https://arxiv.org/ abs/2205.12586. Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. Pedro Saleiro, Benedict Kuester, Abby Stevens, Ari Anisfeld, Loren Hinkson, Jesse London, and Rayid Ghani. Aequitas: A bias and fairness audit toolkit. arXiv preprint arXiv:1811.05577, 2018. Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adri‘a Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022. doi: 10.48550/arXiv.2206.04615. Tensorflow. GitHub - tensorflow/fairness-indicators: Tensorflow’s Fairness Evaluation and Visualization Toolkit, 2020. URL https://github.com/tensorflow/fairness-indicators. Sriram Vasudevan and Krishnaram Kenthapadi. LiFT: A scalable framework for measuring fairness in ml applications. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management, CIKM ’20, 2020. Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, et al. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. 2023. Kellie Webster, Marta Recasens, Vera Axelrod, and Jason Baldridge. Mind the GAP: A balanced corpus of gendered ambiguous pronouns. Transactions of the Association for Computational Linguistics, 6:605–617, 2018. doi: 10.1162/tacl a 00240. URL https: //aclanthology.org/Q18-1042. Hilde Weerts, Miroslav Dudı́k, Richard Edgar, Adrin Jalali, Roman Lutz, and Michael Madaio. Fairlearn: Assessing and Improving Fairness of AI Systems. Journal of Machine Learning Research, 24, 2023. URL http://jmlr.org/papers/v24/23-0389.html. James Wexler, Mahima Pushkarna, Tolga Bolukbasi, Martin Wattenberg, Fernanda B. Viégas, and Jimbo Wilson. The what-if tool: Interactive probing of machine learning models. CoRR, abs/1907.04135, 2019. URL http://arxiv.org/abs/1907.04135. 9

---

<!-- Page 10 -->

Bouchard et al.

Wu Zekun, Sahan Bulathwela, and Adriano Soares Koshiyama. Towards auditing large language models: Improving text-based stereotype detection, 2023. URL https://arxiv. org/abs/2311.14126. Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. Is chatgpt fair for recommendation? evaluating fairness in large language model recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, volume 2012 of RecSys ’23, page 993–999. ACM, September 2023. doi: 10.1145/3604915.3608860. URL http://dx.doi.org/10.1145/3604915.3608860. Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender Bias in Coreference Resolution: Evaluation and Debiasing methods, 4 2018. URL https: //arxiv.org/abs/1804.06876.

10
