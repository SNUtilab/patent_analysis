# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:03:32 2022

version : 1.0

@author: tmlab
"""

# 1. 데이터 로드

if __name__ == '__main__':
    
    import os
    import sys
    import pandas as pd
    import numpy as np     
    import warnings
    
    warnings.filterwarnings("ignore")
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window os
    sys.path.append(directory+'/submodule')
    
    # data 경로설정 필요
    directory = 'D:\원드라이브백업_tilab\TILAB - 문서\DB\patent\wisdomain\포장_유기체'
    directory = directory.replace('\\', "/") # window os
    directory += '/'
    file_list = os.listdir(directory)
    data = pd.DataFrame()
    for file in file_list : 
        temp_data = pd.read_csv(directory + file , skiprows = 4)
        data = pd.concat([data, temp_data], axis = 0).reset_index(drop = 1)
        
    
    #%% 데이터 전처리    
    from preprocess import wisdomain_prep
    
    data_ = wisdomain_prep(data)    
    
    data_sample = data_.loc[data_['year_application'] >= 2015, : ].reset_index(drop = 1) # 데이터 샘플

    
    #%% 1. pre-calculating embeddings
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    embeddings = embedding_model.encode(data_sample['TAF'] ,device='cuda')
    
    
    #%% 2. Preventing Stochastic Behavior
    
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    
    #%% 3. Controlling Number of Topics
    
    from hdbscan import HDBSCAN

    hdbscan_model = HDBSCAN(min_cluster_size=90, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    #%% 4. Improving Default Representation
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    #%% 5. Diversify topic representation
    from bertopic.representation import MaximalMarginalRelevance
    
    representation_model = MaximalMarginalRelevance(diversity=0.5) # default = 0.1
    
    #%% 5. Training
    from bertopic import BERTopic

    topic_model = BERTopic(
    
      # Pipeline models
      embedding_model=embedding_model,
      umap_model=umap_model,
      hdbscan_model=hdbscan_model,
      vectorizer_model=vectorizer_model,
      representation_model=representation_model,
    
      # Hyperparameters
      top_n_words=10,
      verbose=True
    )
    
    # Train model
    topics, probs = topic_model.fit_transform(data_sample['TAF'], embeddings)
    
    temp = topic_model.get_topic_info()
    
    #%% bertopic wrapper 
    
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer
    from hdbscan import HDBSCAN
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic
    from sklearn.cluster import KMeans
    
    docs = data_sample['TAF']
    embedding_model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
    # embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    embeddings = embedding_model.encode(docs ,device='cuda')
    
    def bertopic_wrapper(min_cluister_size, docs) :
    
        umap_model = UMAP(n_neighbors = 15, 
                          n_components = 5, 
                          # min_dist=0.1,
                          metric='cosine', random_state=42) 
        
        # cluster_model = KMeans(n_clusters=min_cluister_size)
        
        cluster_model = HDBSCAN(min_cluster_size= min_cluister_size, 
                                min_samples = 10,                            
                                cluster_selection_method='leaf', 
                                prediction_data=True,
                                gen_min_span_tree=1)
        
        vectorizer_model = CountVectorizer(stop_words="english",
                                           min_df = 2,
                                           ngram_range=(1, 2))
        
        representation_model = MaximalMarginalRelevance(diversity=0.5) # default = 0.1
        
        topic_model = BERTopic(
        
          # Pipeline models
          embedding_model=embedding_model,
          umap_model=umap_model,
          hdbscan_model=cluster_model,
          vectorizer_model=vectorizer_model,
          representation_model=representation_model,
        
          # Hyperparameters
          top_n_words=10,
          verbose=True
        )
        topic_model.fit(docs, embeddings)    
        
        # Train model
        return(topic_model)
    
    
    #%% iter-1
    
    docs = data_sample['TAF']
    
    # min_cluster size에 따른 실루엣 계수 변화
    # from sklearn.metrics import silhouette_score
    
    DBCVs = [] # Density-based clustering validation
    outliers_counts = []
    cluster_N_counts = []
    cluster_big_counts = []
    
    K = range(10, 101, 10)
    
    for size in K :     
        
        topic_model1 = bertopic_wrapper(size, docs)
    
        # result = topic_model1.get_topic_info()
        result_document = topic_model1.get_document_info(docs)
        
        # 아웃라이어 제외 실루엣
        # indexes = list(result_document.loc[result_document['Topic'] != -1, :].index)
        # embeddings_ = embeddings[indexes]
        # label_ = result_document.loc[indexes, 'Topic']
        # score = silhouette_score(embeddings_, label_)
        # silhouette_avgs.append(score)
        
        # dbcv
        score = topic_model1.hdbscan_model.relative_validity_
        
        DBCVs.append(score)
        cluster_N_counts.append(len(topic_model1.topic_sizes_))
        outliers_counts.append(topic_model1.topic_sizes_[-1])
        cluster_big_counts.append(topic_model1.topic_sizes_[0])
        
        
        print(size)
    
    import matplotlib.pyplot as plt
    
    # 시각화
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.plot(K, DBCVs, 'bx-')
    plt.xlabel('min_cluister_size')
    plt.ylabel('DBCVs')
    
    # 시각화
    plt.subplot(222)
    plt.plot(K, outliers_counts, 'rx-')
    plt.xlabel('min_cluister_size')
    plt.ylabel('outliers_counts')
    
    
    # 시각화
    plt.subplot(223)
    plt.plot(K, cluster_N_counts, 'gx-')
    plt.xlabel('min_cluister_size')
    plt.ylabel('cluster_counts')
    
    
    # 시각화
    plt.subplot(224)
    plt.plot(K, cluster_big_counts, 'yx-')
    plt.xlabel('min_cluister_size')
    plt.ylabel('biggest_cluster')
    plt.show()
    
    max_idx = DBCVs.index(max(DBCVs))
    # max_idx = 4|
    
    print("최소 클러스터 사이즈 :", K[max_idx])
    print("최대 DBCVs :", round(DBCVs[max_idx], 3))
    print("아웃라이어 비율 :" , round(outliers_counts[max_idx] / len(docs), 3))
    print("클러스터 수 :" , cluster_N_counts[max_idx])
    print("최대 클러스터의 전체 비율 :" , round(cluster_big_counts[max_idx] / len(docs), 3))
    
    
    #%%
    
    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, 
                          min_dist=0.0, metric='cosine').fit_transform(embeddings)
    
    #%%
    # best model 결과 확인
    import time
    # Train your BERTopic model
    topic_model1 = bertopic_wrapper(K[max_idx], docs)
    
    fig = topic_model1.visualize_documents(docs, reduced_embeddings=reduced_embeddings, 
                                hide_document_hover=True, hide_annotations=True)
    
    directory_ = 'D:/OneDrive - SNU'
    t = time.time()
    time_local = time.localtime(t)
    time_local = time.asctime(time_local).split()[3]
    time_local = time_local.replace(":", "_")
    

    fig.write_html(directory_ + '/visual_test_'+ time_local +'.html')
     
    #%% best model에 대해 outlier reduction 적용

    from bertopic import BERTopic
    
    # Train your BERTopic model
    topic_model1 = bertopic_wrapper(K[max_idx], docs)

    print('아웃라이어 축소 적용 전\n\n',topic_model1.get_topic_freq())
    topics = topic_model1.topics_
    #%%
    # Reduce outliers
    new_topics = topic_model1.reduce_outliers(docs, topics, 
                                              strategy="c-tf-idf", 
                                              threshold = 0.1)

    from collections import Counter
    c = Counter(new_topics)

    print(c)
    
    topic_model1.update_topics(docs, topics = new_topics)
    
    print('아웃라이어 축소 적용 후\n\n',topic_model1.get_topic_freq())
    topics = topic_model1.topics_
    
    
    #%%
    
    
    fig = topic_model1.visualize_documents(docs, reduced_embeddings=reduced_embeddings, 
                                hide_document_hover=True, hide_annotations=True)
    
    directory_ = 'D:/OneDrive - SNU'
    t = time.time()
    time_local = time.localtime(t)
    time_local = time.asctime(time_local).split()[3]
    time_local = time_local.replace(":", "_")
    

    fig.write_html(directory_ + '/visual_test_'+ time_local +'.html')
    
    #%% heatmap
    fig = topic_model1.visualize_heatmap()
    
    t = time.time()
    time_local = time.localtime(t)
    time_local = time.asctime(time_local).split()[3]
    time_local = time_local.replace(":", "_")
    

    fig.write_html(directory_ + '/visual_test_'+ time_local +'.html')
    
    #%% hierarchy
    fig = topic_model1.visualize_hierarchy()
    
    t = time.time()
    time_local = time.localtime(t)
    time_local = time.asctime(time_local).split()[3]
    time_local = time_local.replace(":", "_")
    

    fig.write_html(directory_ + '/visual_test_'+ time_local +'.html')
    
