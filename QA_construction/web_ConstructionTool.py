import os
import json
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from similarity_utils import bge_similarity, rougeL_score
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
# from tool import *
import matplotlib.pyplot as plt
import seaborn as sns

def webRawSelection(rawFilePath: str, singleImgFilePath: str, multiImgFilePath: str):
    """
    Split WebQnA dataset into multiple image facts and single image fact
    """
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    singleImgFacts = {}
    multiImgFacts = {}
    if isinstance(data, dict):
        dataItems = list(data.items())
        for dataItem in dataItems:
            if len(dataItem[1]['img_posFacts']) >= 2:
                del dataItem[1]['img_negFacts']
                del dataItem[1]['txt_negFacts']
                multiImgFacts[dataItem[0]] = dataItem[1]
            else:
                del dataItem[1]['img_negFacts']
                del dataItem[1]['txt_negFacts']
                singleImgFacts[dataItem[0]] = dataItem[1]
    else:
        print("This is not a JSON file!")

    with open(singleImgFilePath, 'w', encoding='utf-8') as f_single:
        json.dump(singleImgFacts, f_single, indent=4, ensure_ascii=False)

    with open(multiImgFilePath, 'w', encoding='utf-8') as f_multi:
        json.dump(multiImgFacts, f_multi, indent=4, ensure_ascii=False)


def webFactsExtraction(webFilePath: str, standardTemplate: str, writeToFilePath: str):
    """
    WebQnA dataset: extract text facts and image facts (only image facts)
    """
    extractionDict = {}
    if webFilePath.endswith('.json'):
        with open(webFilePath, 'r', encoding='utf-8') as f:
            # read and process sample data
            data = json.load(f)
            dataItemList = list(data.items())
            # combine the sample with the prompt template
            i = 0
            for dataItem in dataItemList:
                response = get_chat_completion(standardTemplate.format(dataItem[1]['Q'], str(dataItem[1]['img_Facts'])))
                extractionDict[dataItem[0]] = response[0]
                i += 1
                print("Response completed ", i)
        # write to json file
        write2json(extractionDict, writeToFilePath)
    else:
        print("This is not a JSON file!")


def imageDescriptionGeneration(webFilePath: str, standardTemplate: str, writeToFilePath: str):
    """
    WebQnA dataset: generate image description
    """
    descriptionDict = {}
    if webFilePath.endswith('.json'):
        with open(webFilePath, 'r', encoding='utf-8') as f:
            # read and process sample data
            data = json.load(f)
            dataItemList = list(data.items())
            i = 0
            for dataItem in dataItemList:
                # get all the image facts in a single data item
                descriptionList = []
                imgFacts = dataItem[1]['img_Facts']
                j = 0
                for imgFact in imgFacts:
                    # generate description for each image in a single data item
                    imgDict = {}
                    response = get_chat_completion(
                        standardTemplate.format(imgFact['image_id'], imgFact['title'], imgFact['caption']))
                    imgDict[imgFact['image_id']] = response[0][0]
                    descriptionList.append(imgDict)
                    j += 1
                    print("----- Image description completed ", j)
                descriptionDict[dataItem[0]] = descriptionList
                i += 1
                print("Data item completed ", i)
                # write to json file
        write2json(descriptionDict, writeToFilePath)
    else:
        print("This is not a JSON file!")


def filterData(rawFilePath: str, threshold: float, writeToFilePath: str):
    """
    Filter out data with rouge-L score lower than threshold
    """
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filteredFacts = {}
    progress = 0
    if isinstance(data, dict):
        dataItems = list(data.items())
        for dataItem in dataItems:
            # get all the image facts
            allImgFacts = dataItem[1]['img_posFacts']
            # calculate rouge-L score
            reference_text = allImgFacts[0]['title']
            candidate_text = allImgFacts[1]['title']
            rouge_l = rougeL_score(candidate_text, reference_text)
            if rouge_l <= threshold:
                filteredFacts[dataItem[0]] = dataItem[1]
            progress += 1
            print("Current Progress ", progress)
    write2json(filteredFacts, writeToFilePath)
    print("Filter done")


def addNegText(filterPath: str, rawPath: str, wrtieToFilePath: str):
    with open(rawPath, 'r', encoding='utf-8') as f1:
        dataRaw = json.load(f1)

    with open(filterPath, 'r', encoding='utf-8') as f2:
        dataFilter = json.load(f2)

    dataFilterItems = list(dataFilter.items())
    for dataItem in dataFilterItems:
        dataItem_id = dataItem[0]
        txt_neg = dataRaw[dataItem_id]['txt_negFacts']
        dataItem[1]['txt_negFacts'] = txt_neg
        dataFilter[dataItem_id] = dataItem[1]

    write2json(dataFilter, wrtieToFilePath)
    print("Add negative texts")


def entitiesExtraction(dataItems: Any, extract_Prompt: str, progress: int, actual_progress: int, completeDict: dict,
                       writeToFilePath: str, dataLimit: int):
    """
    loop for extracting entities
    """
    for dataItem in dataItems:
        if progress != dataLimit:
            print("----------------- Progress ", progress, "-----------------")
            progress += 1
            try:
                complete_ExtractTemplate = extract_Prompt.format(dataItem[0], dataItem[1]['Q'])
                extractionResponse = get_chat_completion(complete_ExtractTemplate)
                dataItem[1]['entities'] = extractionResponse[0]['extraction result']
                print("Entities extracted ", actual_progress)
                actual_progress += 1
            except Exception as e:
                print(e)
                continue

            completeDict[dataItem[0]] = dataItem[1]
            write2json(content=completeDict, responseTxtPath=writeToFilePath)
        else:
            break


def entityExtract_Pipeline(rawFilePath: str, extract_Prompt: str, writeToFilePath: str, progress: int,
                           actual_progress: int, dataLimit: int):
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataItems = list(data.items())[progress:dataLimit]

    # overall progress -- progress
    # actual progress -- actual progress
    if os.path.exists(writeToFilePath) and os.stat(writeToFilePath).st_size != 0:
        with open(writeToFilePath, 'r', encoding='utf-8') as f_1:
            existing_data = json.load(f_1)
        entitiesExtraction(dataItems=dataItems, extract_Prompt=extract_Prompt, progress=progress,
                           actual_progress=actual_progress, completeDict=existing_data, writeToFilePath=writeToFilePath,
                           dataLimit=dataLimit)
    else:
        completeDict = {}
        entitiesExtraction(dataItems=dataItems, extract_Prompt=extract_Prompt, progress=progress,
                           actual_progress=actual_progress, completeDict=completeDict, writeToFilePath=writeToFilePath,
                           dataLimit=dataLimit)


def deduplication_entities(rawFilePath: str, filtered_json_file_path: str, removed_json_file_path: str):
    # Set to keep track of unique 'main entity' values
    unique_main_entities = set()
    unique_main_contexts = set()
    filtered_data = {}
    removed_data = {}

    removed_progress_entity = 0
    removed_progress_context = 0
    count = 0
    with open(rawFilePath, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            # Check the 'entities' list for the 'main entity'
            entities = value.get('entities', [])
            imgs = value.get('img_posFacts', [])

            if len(imgs) != 2:
                print("Main entity is not 2 ", count)
                count += 1
                continue

            keep_entry = True
            for entity in entities:
                main_entity = entity.get('main entity')
                choose_context = entity.get('choose')

                # check if the entity is repeated first
                if main_entity in unique_main_entities:
                    keep_entry = False
                    print("removed due to entity overlapped ", removed_progress_entity)
                    removed_progress_entity += 1
                    break
                else:
                    unique_main_entities.add(main_entity)

                # check if the chosen context is repeated second
                if choose_context['snippet_id'] in unique_main_contexts:
                    keep_entry = False
                    print("removed due to context overlapped ", removed_progress_context)
                    removed_progress_context += 1
                    break
                else:
                    unique_main_contexts.add(choose_context['snippet_id'])

            # Only keep the entry if 'main entity' is unique
            if keep_entry:
                filtered_data[key] = value
            else:
                removed_data[key] = value

    # Save the filtered JSON data back to a file
    with open(filtered_json_file_path, 'w') as file:
        json.dump(filtered_data, file, indent=4)

    # Save the removed JSON data back to a file for review
    with open(removed_json_file_path, 'w') as file:
        json.dump(removed_data, file, indent=4)


def contextMatch(entitiesList: list, text_neg: list, model: Any):
    """
    for each entity in each data piece, match the most relevant context, and update the entity list
    """
    update = []
    uniqueSet = set()
    for entity in entitiesList:
        score_list = []
        for text in text_neg:
            # calculate bge score
            score = bge_similarity(text1=entity['main entity'], text2=text['fact'], model=model)
            score_list.append(score)
        max_score = max(score_list)
        max_index = np.argmax(np.array(score_list))
        most_similar_answer = text_neg[max_index]
        most_similar_answer['bge score'] = max_score

        if max_index not in uniqueSet:
            uniqueSet.add(max_index)
        else:
            return

        entity['choose'] = most_similar_answer
        update.append(entity)
    return update


def contextMatch_Pipeline(rawFilePath: str, writeToFilePath: str, progress: int, actual_progress: int, dataLimit: int, imgThreshold: float, model: Any):
    """
    1. use bge score to filter out data with highly relevant image-image; 2. context match using bge score
    """

    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataItems = list(data.items())[progress:dataLimit]

    img_removed = 0
    if os.path.exists(writeToFilePath) and os.stat(writeToFilePath).st_size != 0:
        with open(writeToFilePath, 'r', encoding='utf-8') as f_1:
            existing_data = json.load(f_1)

        for dataItem in dataItems:
            print("----------------- Progress of context match: ", progress, "-----------------")
            progress += 1

            # check if txt_negFacts is empty or not
            if not dataItem[1]['txt_negFacts']:
                continue

            # filter out highly relevant img-img data
            imgFacts = dataItem[1]['img_posFacts']
            img_imgScore = bge_similarity(imgFacts[0]['title'], imgFacts[1]['title'], model=model)
            if img_imgScore > imgThreshold:
                print(img_removed, " image-image removed due to high relevance")
                img_removed += 1
                continue

            update = contextMatch(entitiesList=dataItem[1]['entities'], text_neg=dataItem[1]['txt_negFacts'], model=model)

            if update:
                dataItem[1]['entities'] = update
                existing_data[dataItem[0]] = dataItem[1]
                print("Actual progress ", actual_progress)
                actual_progress += 1
            else:
                continue
            write2json(content=existing_data, responseTxtPath=writeToFilePath)

    else:
        completeDict = {}

        for dataItem in dataItems:
            print("----------------- Progress of context match: ", progress, "-----------------")
            progress += 1

            # check if txt_negFacts is empty or not
            if not dataItem[1]['txt_negFacts']:
                continue

            # filter out highly relevant img-img data
            imgFacts = dataItem[1]['img_posFacts']
            img_imgScore = bge_similarity(imgFacts[0]['title'], imgFacts[1]['title'], model=model)
            if img_imgScore > imgThreshold:
                print(img_removed, " image-image removed due to high relevance")
                img_removed += 1
                continue

            update = contextMatch(entitiesList=dataItem[1]['entities'], text_neg=dataItem[1]['txt_negFacts'], model=model)

            if update:
                dataItem[1]['entities'] = update
                completeDict[dataItem[0]] = dataItem[1]
                print("Actual progress ", actual_progress)
                actual_progress += 1
            else:
                continue
            write2json(content=completeDict, responseTxtPath=writeToFilePath)


def ensureMultiImg(rawFilePath: str, writeToPath: str):
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataItems = list(data.items())

    finalDict = {}
    count = 1
    single_count = 1
    for dataItem in dataItems:
        imgPos = dataItem[1]['img_posFacts']
        if len(imgPos) > 2:
            print(count, " data piece with more than 2 images")
            count += 1

        if len(imgPos) == 1:
            print(single_count, " data piece with only one image")
            single_count += 1

        if len(imgPos) == 2:
            finalDict[dataItem[0]] = dataItem[1]
    write2json(content=finalDict, responseTxtPath=writeToPath)


def Draw_bgeScore(rawFilePath: str, dataLimit: int):
    """
    Draw bge score distribution figure
    """
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataItems = list(data.items())

    bge_score = []
    count = 0
    for dataItem in dataItems:
        if count != dataLimit:
            entities = dataItem[1]['entities']

            for entity in entities:

                bge_score.append(entity['choose']['bge score'])

            count += 1
        else:
            break
    # 设置图形风格
    sns.set(style="whitegrid")

    # 将list转换为NumPy数组
    bge_score = np.array(bge_score)

    # 创建直方图
    plt.figure(figsize=(10, 6))
    plt.hist(bge_score, bins=10, color="blue", edgecolor="black")

    # 添加标题和标签
    plt.title('BGE Scores Distribution', fontsize=15)
    plt.xlabel('BGE Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 显示图形
    plt.show()


def removeByBGE(rawFilePath: str, dataLimit: int, entityThreashold: float, writeToFile: str):
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataItems = list(data.items())

    finalDict = {}
    count = 0 # current progress
    keptData = 0 # how many data is kept
    context_removed = 0
    for dataItem in dataItems:
        if count != dataLimit:
            keepEntry = True
            entities = dataItem[1]['entities']

            for entity in entities:
                entity_score = entity['choose']['bge score']
                if entity_score < entityThreashold:
                    print("Entity-context no relevance removed ", context_removed)
                    context_removed += 1
                    keepEntry = False

            if keepEntry:
                finalDict[dataItem[0]] = dataItem[1]
                print("============ Data kept ", keptData, "============")
                keptData += 1
            write2json(content=finalDict, responseTxtPath=writeToFile)
            count += 1
        else:
            break


if __name__ == '__main__':
    # 如果还没有下载 NLTK 的 punkt 数据集，请运行以下命令
    # nltk.download('punkt')
    # nltk.download('punkt_tab')

    # 加载 BAAI/bge-base-en-v1.5 模型
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')

    # =============================================== WebQnA
    # Contexts and images extraction
    webExtractionTemplate = """\
        You are a professor, skilled at finding answers from text. Based on the content of the question (Q) in the following Original JSON, please select the facts from text_facts and image_facts that are truly likely to answer the question. Return the original JSON format.
        Requirements:
        1. The selected text facts and image facts should be at least more than two.           
        2. The selected text facts and image facts should remain original json format.

        Original JSON: {}
        ---------------------
        Output Format (Please strictly follow this format and do not add any extra text):
        ```json
        [
            "question": "",
            "txt_Facts": ["title", "fact", "url", "snippet_id"],
            "img_Facts": ["image_id", "title", "caption", "url", "imgUrl"]
        ]

        """

    webExtractionTemplate2 = """\
        You are a professor, skilled at finding evidences from texts and images. Based on the text content of the provided question (Q) and provided list of image facts, you are required to finding evidences.
        ----------------------
        Question: {}
        Image Facts: {}
        ----------------------
        Please select the facts from the provided list of image facts that are truly likely to answer the question. Return the original JSON format. Please carefully compare the question with all the information in the image_facts, and determine whether it truly helps answer the question. Choose at least two image facts.
        Requirements:
        1.The image facts selected should be more than 2.
        2.For output, remain the original Image Fact JSON format as it is in the provided JSON dict.
        3.The output must be in a json dict, with the following structure:
        ```json
        [
        	"Question": ' ',
        	"img_Facts": [ ]
        ]
        """

    entitiesExtractionTemplate = """\
    Question information is below. 
    ---------------------
    Question id:{}
    Qustion:{}
    You are a professor skilled at identifying the main entities and core aspects (entity + feature) of a question.
    Please analyze the provided question and extract the main entities and the core aspects (entity + feature) that are being inquired about. Each question will involve two entities and two core aspects (entity + feature).
    Example:
    ```json
    [
        "question id": "30",
        "question": "Are both the National Museum of the American Indian in Washington, D.C. and the Xanadu House in Kissimmee, Florida the same color?",
        "extraction result": [
            ```json
            [
                "main entity": "National Museum of the American Indian in Washington, D.C.",
                "entity feature": "The colors of the National Museum of the American Indian in Washington, D.C."
            ],
            ```
            ```json
            [
                "main entity": "Xanadu House in Kissimmee, Florida",
                "entity feature": "The colors of the Xanadu House in Kissimmee, Florida"
            ]
            ```
        ]
    ],
    ```
    ```json
    [
        "question id": "20",
        "question": "What Russian architectural feature can be found on both the Assumption Cathedral in Vladimir and Saint Basil's Cathedral in Moscow?",
        "extraction result": [
            ```json
            [
                "main entity": "Assumption Cathedral in Vladimir",
                "entity feature": "Russian architectural feature of the Assumption Cathedral in Vladimir"
            ],
            ```
            ```json
            [
                "main entity": "Saint Basil's Cathedral in Moscow",
                "entity feature": "Russian architectural feature of Saint Basil's Cathedral in Moscow"
            ]
            ```
        ]
    ]
    ```
    For each question, the output should be in the following JSON format:
    ```json
        "question id": " ",
        "extraction result": [
            ```json
            [
                "main entity": "",
                "entity feature": ""
            ],
            ```
            ```json
            [
                "main entity": "",
                "entity feature": ""
            ]
            ```
        ]
    ```
    """

    imgDescriptionTemplate = """\
        Picture information is below. 
        --------------------- 
        Picture ID: {}
        Picture Title: {} 
        Picture Caption: {}
        --------------------- 
        Please generate a detailed description for the picture. You are a Teacher/Professor. Describe the overall setting, environment, or location, providing context to the background.Clearly identify the main subject of the image, providing details about their appearance, posture, and specific characteristics. 
        Requirements:
        1.In the generated, the picture should be inserted in the appropriate position. '[Picture ID]' is used to represent the picture and 
        you are required to insert '[Picture ID]' in appropriate positions. For instance, 'The picture showcases [30090059] the Original Playboy Mansion, located at 1340 N State Parkway in Chicago, Illinois. The mansion is an iconic architectural gem, characterized by its grand façade featuring intricate detailing and a classic brick exterior. The setting is lush, with well-manicured gardens and mature trees framing the property, which provide a serene contrast to the bustling city backdrop. The atmosphere evokes a sense of nostalgia and opulence, reminiscent of the lavish parties once held within its walls. \n\nThe main subject of the image is the mansion itself, which stands prominently as a historical landmark. Its posture is one of elegance and strength, with tall windows and ornate balconies that suggest a vibrant social history. The roofline is distinctive, often highlighted by decorative elements that speak to the era in which it was built. Although there are no people depicted in the image, one can almost imagine the glamorous social gatherings that took place here, filled with laughter and lively conversation. Overall, the image captures the essence of a bygone era of luxury and allure associated with the Playboy brand.'
        Output Format:
        [```json
            "description"
        ]
        ```
        """

    # Overall Pipeline:
    # 1. split raw dataset WebQA_train_val.json into single-image json and multiple-image json
    # 2. entities extraction on multiple-image json
    # 3. context match on multiple-image json
    # 4. deduplication according to entities and context

    RawFilePath = '../WebQnA/WebQA_data_first_release/WebQA_train_val.json'
    multiPath = 'dataProcessed/web_multiImg_filtered_unique_full.json'
    singlePath = 'dataProcessed/web_singleImg_full.json'

    entitiesExtractedPath1 = 'dataProcessed/web_multi_entities_full_1.json'
    entitiesExtractedPath2 = 'dataProcessed/web_multi_entities_full_2.json'

    matchedFilePath = 'dataProcessed/web_multi_entities_matched_bge.json'
    matchedFiltered = 'dataProcessed/web_multi_matched_FILTERED_bge.json'
    matchedRemoved = 'dataProcessed/web_multi_matched_REMOVED_bge.json'

    relevanceFiltered = 'dataProcessed/web_multi_matched_FILTERED_bge_rel-0.65.json'

    try:
        # ======= split raw dataset =======
        # finished

        # ======= entities extraction ======= (gpt involved)
        # progress = 2031
        # actual_progress = 1704
        # dataLimit = 4020
        # entityExtract_Pipeline(rawFilePath=multiPath, extract_Prompt=entitiesExtractionTemplate,
        #                        writeToFilePath=entitiesExtractedPath,
        #                        progress=progress, actual_progress=actual_progress, dataLimit=dataLimit)

        # ======= context matching =======
        # progress = 0
        # actual_progress = 1158
        # dataLimit = 2000
        # contextMatch_Pipeline(rawFilePath=entitiesExtractedPath2, writeToFilePath=matchedFilePath, progress=progress,
        #                       actual_progress=actual_progress, dataLimit=dataLimit, imgThreshold=0.7, model=model)
        # print("ok")

        # ======= deduplication =======
        # ensureMultiImg(rawFilePath=matchedFiltered, writeToPath=matchedFiltered)
        # deduplication_entities(rawFilePath=matchedFilePath, filtered_json_file_path=matchedFiltered, removed_json_file_path=matchedRemoved)
        removeByBGE(rawFilePath=matchedFiltered, dataLimit=2300, entityThreashold=0.65, writeToFile=relevanceFiltered)
        # Draw_bgeScore(rawFilePath=matchedFilePath, dataLimit=2300)
    except Exception as e:
        logging.critical(f"Failed during main execution: {e}")