from tool import *
import pandas as pd
import requests


def wit_MultipleSamples(count: int, countStop: int, contexts: list, pictureIds: list, pictureCaps: list, standardTemplate: str):
    """
    Automatically process multiple samples in WIT
    """
    i = 0
    finalDict = {}
    while count != countStop:
        response = get_chat_completion(standardTemplate.format(contexts[i], pictureIds[i], pictureCaps[i]))
        finalDict[count] = response[0]
        count += 1
        i += 1
        print("Response completed ", i)
    write2json(finalDict, 'response32-49.json')
    print("Finished")


def wit_dataSelection(rawFilePath: str):
    df = pd.read_csv(rawFilePath, delimiter='\t')
    # selection requirements
    df_filtered = df[(df['is_main_image'] == True) & (df['language'] == 'en') & (df['caption_reference_description'].notna()) & (df['caption_attribution_description'].notna()) & (df['page_changed_recently'] == False) & (df['context_page_description'].str.len() >= 200) & (df['original_height'] <= 3500) & (df['original_width'] <= 3500) & (df['mime_type'] == 'image/jpeg') ]
    return df_filtered


def combine_dataframes(dfs: list, target_rows: int, writeToFilePath: str):
    """
    根据输入的DataFrame列表和目标行数，拼接多个DataFrame
    dfs: list of DataFrames
    target_rows: 最终合成的DataFrame中期望的行数
    """
    combined_df = pd.DataFrame()  # 用于存储合并后的DataFrame
    current_rows = 0  # 当前累积的行数

    # 遍历输入的DataFrame列表
    for df in dfs:
        # 如果当前的累积行数已经达到或超过目标行数，停止合并
        if current_rows >= target_rows:
            break

        # 需要从当前DataFrame中选取的行数
        rows_needed = target_rows - current_rows

        # 如果当前DataFrame行数大于或等于所需行数，则只取部分
        if len(df) > rows_needed:
            df = df.head(rows_needed)

        # 将选定的部分追加到合并的DataFrame中
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # 更新当前的累积行数
        current_rows = len(combined_df)

    # 添加context id 和 image id
    combined_df['image_id'] = None
    combined_df['context_id'] = None
    ImgID = "2_"
    ContextID = 20000
    for i in range(len(combined_df)):
        combined_df.at[i, 'image_id'] = ImgID + str(i)
        combined_df.at[i, 'context_id'] = ContextID + i
    combined_df.to_csv(writeToFilePath, sep='\t', index=False)


def qa_OneSample(dataFrame: Any, q_Prompt: str, a_Prompt: str):
    questionCompleted = q_Prompt.format(
        dataFrame['context_page_description'],
        dataFrame['caption_reference_description'],
        dataFrame['image_id']
    )
    questionResponse = get_chat_completion(questionCompleted)

    answerCompleted = a_Prompt.format(
        dataFrame['context_page_description'],
        dataFrame['image_id'],
        str(questionResponse[0])
    )

    finalResponse = get_chat_completion(answerCompleted)
    return finalResponse[0]


def qa_Generation(rawFilePath: str, q_Prompt: str, a_Prompt: str, r_Prompt: str, writeToFilePath: str, dataLimit: int, startIndex: int, startIndex_all: int):
    """
    Input raw tsv file, process each row in dataframe using GPT
    startIndex: index of the current data already written into json file
    startIndex_all: index of the current data (progress) in the overall process
    """
    df_raw = pd.read_csv(rawFilePath, delimiter='\t')
    df = df_raw.iloc[startIndex_all:] # overall progress
    # iterate each row in dataframe
    if os.path.exists(writeToFilePath) and os.stat(writeToFilePath).st_size != 0:
        with open(writeToFilePath, 'r', encoding='utf-8') as f_1:
            existing_data = json.load(f_1)
        wit_GenerationPipeline(startIndex=startIndex, startIndex_all=startIndex_all, dataLimit=dataLimit, df=df, a_Prompt=a_Prompt,
                                   q_Prompt=q_Prompt, r_Prompt=r_Prompt, completeDict=existing_data, writeToFilePath=writeToFilePath)
    else:
        completeDict = {}
        wit_GenerationPipeline(startIndex=startIndex, startIndex_all=startIndex_all, dataLimit=dataLimit, df=df, a_Prompt=a_Prompt,
                                   q_Prompt=q_Prompt, r_Prompt=r_Prompt, completeDict=completeDict, writeToFilePath=writeToFilePath)


def wit_GenerationPipeline(startIndex: int, dataLimit: int, df: Any, r_Prompt: str, q_Prompt: str, a_Prompt: str, completeDict: dict, writeToFilePath: str, startIndex_all: int):
    for index, row in df.iterrows():
        if index != dataLimit:
            print("------------- Progress ", startIndex_all, "-------------")
            startIndex_all += 1

            # step 1: relevance determination "yes" or "no"?
            try:
                r_response = relevanceSelection(dataFrame=row, r_Prompt=r_Prompt)
            except Exception as e:
                print("Error: ", e)
                continue


            if r_response['Relevance'] == "no":
                continue
            print("Relevance determination pass ", startIndex)

            # step 2: question and answer generation
            qaList = qa_OneSample(dataFrame=row, q_Prompt=q_Prompt, a_Prompt=a_Prompt)

            # step 3: question validation determine
            qa_valid_dict = {}
            qa_index = 0
            for qa in qaList:
                if qa['Valid'] == "no":
                    continue
                qa_valid_dict[qa_index] = qa
                qa_index += 1
            if not qa_valid_dict:
                continue

            sampleDict = {
                "Context": row['context_page_description'],
                "Context ID": row['context_id'],
                "Context url": row['page_url'],
                "Image Caption": row['caption_reference_description'],
                "Image Description": row['caption_attribution_description'],
                "Image ID": row['image_id'],
                "Image url": row['image_url'],
                "QA List": qa_valid_dict
            }

            data_id = "02_" + str(startIndex)
            completeDict[data_id] = sampleDict
            # dump json into file
            write2json(content=completeDict, responseTxtPath=writeToFilePath)
            print("QA Generation complete ", startIndex)
            startIndex += 1
        else:
            break


def relevanceSelection(dataFrame: Any, r_Prompt: str):
    relevanceCompleted = r_Prompt.format(
        dataFrame['context_page_description'],
        dataFrame['caption_reference_description'],
        dataFrame['image_id'],
        dataFrame['context_id'],
    )
    image_part = [dataFrame['image_url']]
    relevanceResponse = get_vision_chat_completion(chat=relevanceCompleted, image_urls=image_part)
    return relevanceResponse[0]


if __name__ == '__main__':
    # =============================================== WIT

    questionGenerationPrompt = """\
    Context and Picture information is below. 
    ---------------------
    Context: {}
    Picture caption: {}
    Picture id: {}
    ---------------------
    Given the context and picture information and not prior knowledge, generate four question based on the query below. 
    You are tasked with generating high-quality evaluation questions for Retrieval-Augmented Generation (RAG) systems, where the image plays a crucial role in forming the answer. The questions should be designed so that the image becomes an indispensable component of the response, and can be answered by combining information from both the text and the image.
    
    Requirements: 
    1.Each question must be self-contained, without referencing terms like "context."  The questions should be framed based on the content of the image. Questions should implicitly relate to the content of the picture, allowing the answer to naturally reference or involve the picture.  The image must play an essential role in the answer, without being directly mentioned in the question. 
    2.The image must be an essential part of the answer, aiding in understanding.  The questioner is unaware of the image beforehand, and the image should clarify the answer.  Carefully consider where to insert the picture ID. Replace pronouns like "this" or "the" with complete descriptions relevant to the text. 
    3.Answers must strictly use the provided content, with no external information introduced.  Only minimal edits for clarity are allowed.  Do not mention 'image' in the answer, insert the image ID in the appropriate places instead."
    4.Thoroughly review each question to ensure it meets the above criteria, as this will impact your compensation.For example, if the image is a portrait and the question asks "when," this would be an invalid QA.The answer must insert the picture ID.
    5.Particularly, the generated questions and answers should not explicitly reference the image or the caption of the image.
    Example: 
    Example Context: "Peacock Sound is an ice-filled sound, 216 kilometres long and 64 km wide, separating Thurston Island from the Eights Coast of Ellsworth Land in Antarctica. The sound is occupied by the western part of the Abbot Ice Shelf, and is therefore not navigable by ships. [pic_1]" 
    Example Picture Caption: "Location of Peacock Sound in Antarctica"
    Example Generated 
    [ " Why is Peacock Sound not navigable by ships?",
     "Peacock Sound [pic_1] is not navigable by ships because it is occupied by the western part of the Abbot Ice Shelf. 
    " 
    ]
    
    Output Format:
    ```json
    [
    
        1. "question":" "  ,
        2. "question":" "  ,
        ...
    ]
    ```
    """

    answerGenerationPrompt = """\
    Context and Picture information is below. 
    ---------------------
    Context: {}
    Picture id: {}
    Questions:{}
    ---------------------
    Given the context and questions, and without relying on prior knowledge, generate an answer for each question.
    You are a professional question-answering expert. Follow these steps carefully to provide a structured and evidence-based answer for each question, where the image plays a crucial role in forming the answer. The questions should be designed so that the image becomes an indispensable component of the response, and can be answered by combining information from both the text and the image.
    
    Steps: 
    1.Check Question Validity:Assess whether the provided question can be answered directly using the context. If the context does not contain any sentence that can be used to answer the question, label it as "Invalid" (No). If evidence can be found within the context, label the question as "Valid" (Yes). If the question is invalid, leave the rest of the task blank.
    2.For Valid Questions(yes): Find Evidence:  Identify the key evidence within the original context that directly supports the answer. The evidence must come strictly from the context; no external information should be added. Extract the main keywords from the evidence that are essential for forming the answer.
    3.Generate Answer:Based on the question, evidence, and keywords, generate a precise answer. Ensure that the answer is directly aligned with the evidence and includes the keywords for consistency, without introducing any external information. Only minimal edits for clarity are permitted.
    4.Insert Image ID:Insert the image ID (e.g., [3_0.jpg]) at the end of the answer. If the image is not strongly relevant to both the question and the answer, consider marking the question as invalid. Make sure the image ID adds value and relevance to the answer, enhancing clarity or providing visual context.
    
    Example: 
    Example Context: "Peacock Sound is an ice-filled sound, 216 kilometres long and 64 km wide, separating Thurston Island from the Eights Coast of Ellsworth Land in Antarctica. The sound is occupied by the western part of the Abbot Ice Shelf, and is therefore not navigable by ships. [pic_1]" 
    Good QA:
    [```json
        "Question": "Why is Peacock Sound not navigable by ships?",
        "Valid": "yes",
        "Evidence": "The sound is occupied by the western part of the Abbot Ice Shelf, and is therefore not navigable by ships.",
        "Keyword": "occupied by the western part of the Abbot Ice Shelf",
        "Answer": "Peacock Sound  is not navigable by ships because it is occupied by the western part of the Abbot Ice Shelf.[pic_1]" 
    ```]
    
    Example Context: "Rosina Emmet Sherwood [pic_2] was an American painter."
    Bad QA:
    [```json 
        "Question": "What is the time period during which Rosina Emmet Sherwood was active as an American painter?", 
        "Valid": "no", 
        "Evidence': " ", 
        "keyword": " ", 
        "Answer": " "
    ```]
    
    Output in json Format:
    ```json
    [
        1. "Question":" "  ,"Valid":" " ,"Evidence":" " ,"keyword":" ", ""Answer":" " ,
        2. "Question":" "  ,"Valid":" " ,"Evidence":" " ,"keyword":" ","Answer":" " ,
        ...
    ]
    ```
    """

    relevanceSelectionPrompt = """\
    Context and Picture information is below. 
    ---------------------
    Context: {}
    Picture caption: {}
    Picture id: {}
    Context id: {}
    ---------------------
    You are an evaluator tasked with determining the relevance between images and text.
    Your task is to assess whether each image is strongly relevant to the provided text.
    
    Requirements:
    1.If an image is only related to an unimportant entity within the text, it should be marked as "Not Qualified(no)."
    2.The image must align with the main theme of the text and provide support for understanding the content.
    3.Images that genuinely assist in understanding the text should be considered "Qualified(yes)." For example, when introducing a person, providing a photo would be considered helpful. Please provide a justification for the decision by giving a "reason."
    
    The following are examples:
    [
        [
        ```json
            "context": "Jaktorów is a village in Grodzisk Mazowiecki County, Masovian Voivodeship, in east-central Poland. It is the seat of the gmina called Gmina Jaktorów. It lies approximately 8 kilometres west of Grodzisk Mazowiecki and 37 km southwest of Warsaw. The village has a population of 910. The last recorded aurochs, a female, died in 1627 in the Jaktorów Forest, Poland. Also called the urus, aurochs were the ancestors of domestic cattle, inhabiting Europe, Asia, and North Africa. The skull of the last recorded specimen was later stolen by the Swedish Army during the Swedish invasion of Poland and is now in Livrustkammaren in Stockholm.",
            "image_caption": "Monument to the last aurochs",
            "Relevance": "no",
            "Reason":"The image of the monument does not provide significant additional understanding of the main text, which discusses the village, its history, and the historical facts surrounding the aurochs. While the monument is indirectly connected to the aurochs mentioned, it is not essential for understanding the main content."
        ```
        ],
        [
        ```json
            "context": "Annepont is a commune in the Charente-Maritime department in the Nouvelle-Aquitaine region of southwestern France. The inhabitants of the commune are known as Annepontois or Annepontoises.",
            "image_caption": "Watermill",
            "Relevance": "no",
            "Reason":"The watermill is not relevant to the core content of the text, which focuses on the commune of Annepont and its inhabitants. The image of a watermill does not aid in understanding the commune or its identity."
        ```
        ],
        [
        ```json
            "context": "The Women's Sports Foundation is an educational nonprofit charity focused on promoting female participation in sports. It was founded in 1974 by tennis player Billie Jean King, with early support from Olympic athletes Donna de Varona and Suzy Chaffee. The foundation's mission is 'To advance the lives of girls and women through sports and physical activity.'",
            "image_caption": "WomensSportsFoundationUpdatedLogo",
            "Relevance": "yes"
            "Reason": "The image of the logo is relevant because it provides a visual identifier for the organization being discussed, helping readers associate the foundation with its mission and history, thereby aiding in understanding."
        ```
        ],
        [
        ```json
            "Context ": "Nolana is a genus of hard annual or perennial plants in the nightshade family. The genus is mostly native to Chile and Peru. Species in this genus, especially N. paradoxa, serve as a model system for studies on flower color.",
            "Image Caption": "Nolana",
            "Relevance": "yes",
            "Reason": "The image is relevant as it depicts 'Nolana,' the main subject in the text. The genus is central to the context, and showing the flower’s appearance enhances understanding, especially since the text discusses the genus's importance in studies related to flower color."
        ```
        ]
    ]
    
    Output in json Format:
    ```json
        "Context ID": " ", 
        "Image ID": " ", 
        "Relevance": " ", 
        "Reason": " "
    ```
    """
    # raw1 = '../WIT/Validation/wit_v1.val.all-00000-of-00005.tsv'
    # raw2 = '../WIT/Validation/wit_v1.val.all-00001-of-00005.tsv'
    # raw3 = '../WIT/Validation/wit_v1.val.all-00002-of-00005.tsv'
    # raw4 = '../WIT/Validation/wit_v1.val.all-00003-of-00005.tsv'
    # raw5 = '../WIT/Validation/wit_v1.val.all-00004-of-00005.tsv'
    # raw6 = '../WIT/Test/wit_v1.test.all-00000-of-00005.tsv'
    # df1 = wit_dataSelection(raw1)
    # df2 = wit_dataSelection(raw2)
    # df3 = wit_dataSelection(raw3)
    # df4 = wit_dataSelection(raw4)
    # df5 = wit_dataSelection(raw5)
    # df6 = wit_dataSelection(raw6)
    # selectionPath = 'dataProcessed/wit_selection1500.tsv'
    # combine_dataframes([df1, df2, df3, df4, df5, df6], 1500, selectionPath)
    # print("yes")

    try:
        # ======================= QA one sample generation
        # df = pd.read_csv('dataProcessed/wit_selection1500.tsv', delimiter='\t')
        # first = df.iloc[152]
        # relevanceResponse = relevanceSelection(dataFrame=first, r_Prompt=relevanceSelectionPrompt)
        # finalResponse = qa_OneSample(dataFrame=first, q_Prompt=questionGenerationPrompt, a_Prompt=answerGenerationPrompt)
        #
        # print("Complete Response: ", finalResponse)

        # ======================= QA multiple samples generation
        rawFilePath = 'dataProcessed/wit_selection1500.tsv'
        finalFilePath = 'dataProcessed/wit_Complete/wit_init_1.json'
        startIndex = 416
        # --- set range of data to be processed
        startIndex_all = 482
        dataLimit = 1300 # index upper limit
        qa_Generation(startIndex=startIndex, startIndex_all=startIndex_all, rawFilePath=rawFilePath,
                      q_Prompt=questionGenerationPrompt, a_Prompt=answerGenerationPrompt, r_Prompt=relevanceSelectionPrompt, writeToFilePath=finalFilePath, dataLimit=dataLimit)
        print("All the generations are completed!!")
    except Exception as e:
        logging.critical(f"Failed during main execution: {e}")