from tool import *
import re
import base64


def dataSelection(folderPath: str, outputFile: str, selectNumber: int):
    # 用于存储所有 JSON 文件内容的字典
    combined_data = {}
    count = 0
    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(folderPath):
        if filename.endswith('.json') and count != selectNumber:
            file_path = os.path.join(folderPath, filename)

            # 读取 JSON 文件的内容
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                steps = data['steps']
                overallImg = 0
                for step in steps:
                    overallImg += len(step['images'])
                    step['images_num'] = len(step['images'])
                data['steps'] = steps
                data['overallImg_num'] = overallImg
                dict_name = "recipeStepsEn_" + str(count)

                # 使用文件名作为字典的键，存储内容
                if overallImg > 2:
                    combined_data[dict_name] = data
                    count += 1
                    print("File iterate complete ", count)
    # 将合并的内容写入到新的 JSON 文件中
    with open(outputFile, 'w', encoding='utf-8') as f_out:
        json.dump(combined_data, f_out, indent=4, ensure_ascii=False)

    print(f"已将 {len(combined_data)} 个 JSON 文件的内容写入 {outputFile}")


def QnAGeneration(rawFilePath: str, standardPrompt: str, writeToPath: str):
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    QnADict = {}
    if isinstance(data, dict):
        dataItems = list(data.items())
        i = 0
        for dataItem in dataItems:
            response = get_chat_completion(standardPrompt.format(str(dataItem)))
            QnADict[dataItem[0]] = response[0]
            i += 1
            print("Q-A generated ", i)
    write2json(content=QnADict, responseTxtPath=writeToPath)


def renewURL(rawFilePath: str, writeToPath: str):
    with open(rawFilePath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    renewDict = {}
    if isinstance(data, dict):
        dataItems = list(data.items())
        i = 0
        for dataItem in dataItems:
            steps = dataItem[1]['steps']
            for step in steps:
                imgURLs = step['image_urls']
                new_urls = [re.sub(r'\.\w+\.jpg', '.jpg', url.replace("cdn", "content")) for url in imgURLs]
                step['image_urls'] = new_urls
            dataItem[1]['steps'] = steps
            renewDict[dataItem[0]] = dataItem[1]
            i += 1
            print("Q-A renewed ", i)
    write2json(content=renewDict, responseTxtPath=writeToPath)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


if __name__ == '__main__':
    QnA_generationTemplate = '''\
    Recipe steps context and corresponding images in each step are provided below.
    ---------------------
    {}
    ---------------------
    Given the specific steps and images information for each step without prior knowledge, generate a question and answer.
    Requirements:
    1. The generated question and answer should be closely related to the provided steps contexts and images.
    2. The generated answer should be image-text interleaved. For instance, 'Step 5: Adding the Cookie Dough to the Pan. First, grease the pan with Pam or a similar oil spray to prevent the dessert from sticking [oreo-cookie-brownies_5_0.jpg]. Then smush the cookie dough into the bottom of your 13x9 pan. Make sure the cookie dough covers the entire surface. [oreo-cookie-brownies_5_1.jpg]'
    3. Generate three question-answer dict pairs in each "QA" dict.
    4. The output should strictly follow the format:
    Output format:
    [```json
        "recipe_id": ' ',
        "QA": [
            [```json
                "Question": ' ', "Answer": ' '
            ],
            [```json
            "Question": ' ', "Answer": ' '
            ],
            [```json
            "Question": ' ', "Answer": ' '
            ]
        ]
    ]
    '''

    Picture_prompt = """\
    Recipe steps context is provided below.
    ---------------------
    {}
    ---------------------

    As a professional chef and scene illustrator, please generate detailed descriptions of the images in URLs involved in the cooking process based on the recipe steps. Carefully depict what is happening in each image to help the viewer better understand the cooking process.
    Output Format:
    [```json
        "image_url": ' '
        "description": ' '
    ]
    ```
    """

    # rawFilePath = '../recipeDataEN/recipeEN_selected_10_v1.json'
    # write2File = 'dataProcessed/recipeEN_selected_10_renewURL.json'
    # renewURL(rawFilePath=rawFilePath, writeToPath=write2File)


    # folder_path = '../recipeDataEN/recipes/recipes-val/steps'
    # write_path = '../recipeDataEN/recipeEN_selected_10_v1.json'
    # dataSelection(folderPath=folder_path, outputFile=write_path, selectNumber=10)
    try:
        # rawFile = '../recipeDataEN/recipeEN_selected_10.json'
        # toFile = 'dataProcessed/recipeQA.json'
        # QnAGeneration(rawFilePath=rawFile, standardPrompt=QnA_generationTemplate, writeToPath=toFile)
        step = "Step 10: Taking the Oreo Cookie Brownies Out of the Oven. Remove the dessert from the oven once the timer is up. Be sure to protect your hands from the heat!"
        imageURLs = ["https://content.instructables.com/FPI/C01J/I9JRDRPP/FPIC01JI9JRDRPP.jpg"]
        response = get_vision_chat_completion(Picture_prompt.format(step), imageURLs)
        print("完整响应内容:", response)
    except Exception as e:
        logging.critical(f"Failed during main execution: {e}")