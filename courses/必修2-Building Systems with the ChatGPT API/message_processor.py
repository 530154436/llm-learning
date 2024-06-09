#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from typing import List, Tuple

import tools
import prompts
from src.utils.chat_robot import OpenAiChat

# 提示词
DELIMITER = "```"
FIND_CATEGORY_AND_PRODUCT_PROMPT = prompts.load_prompt_find_category_and_product(delimiter=DELIMITER)
ASSISTANT_PROMPT = prompts.load_prompt("data/1_assistant.tmpl")
EVALUATION_PROMPT = prompts.load_prompt("data/2_evaluation.tmpl").format(delimiter=DELIMITER)


def find_category_and_product(user_input: str, delimiter=DELIMITER):
    """
    从用户问题中抽取商品和类别
    :param user_input: 用户消息
    :param delimiter: 分隔符
    :return:
    """
    messages = [
        {'role': 'system', 'content': FIND_CATEGORY_AND_PRODUCT_PROMPT},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}\n"
                                    f"返回一个json对象的列表字符串，不要包含任何其他无关的字符（含分隔符）。"},
    ]
    # print(messages)
    response = OpenAiChat().get_completion_from_messages(messages)
    return response


def assistant(user_input: str, product_info: str) -> Tuple[list, str]:
    """
    根据信息生成回答
    :param user_input: 用户输入
    :param product_info: 产品信息
    :return:
    """
    messages = [
        {'role': 'system', 'content': ASSISTANT_PROMPT},
        {'role': 'user', 'content': user_input},
        {'role': 'assistant', 'content': f"""相关产品信息:\n{product_info}"""},
    ]
    response = OpenAiChat().get_completion_from_messages(messages)
    return messages, response


def evaluation(user_input: str, response: str, delimiter=DELIMITER) -> str:
    """
    对生成的回答进行评估
    :param user_input: 用户输入
    :param response: 生成的回答
     :param delimiter:
    :return:
    """
    user_message = f"用户信息: {delimiter}{user_input}{delimiter}\n" \
                   f"代理回复: {delimiter}{response}{delimiter}\n" \
                   f"回复是否正确使用了产品的信息？\n" \
                   f"回复是否充分地回答了问题？\n" \
                   f"返回一个json对象，不要包含任何其他无关的字符（含分隔符）。"
    messages = [
        {'role': 'system', 'content': EVALUATION_PROMPT},
        {'role': 'user', 'content': user_message}
    ]
    response = OpenAiChat().get_completion_from_messages(messages)
    return response


def process_message(user_input: str, context: List[dict], debug=True) -> str:
    """
    对用户信息进行处理

    参数:
    user_input : 用户输入
    context : 历史信息
    debug : 是否开启 DEBUG 模式,默认开启
    """
    # 分隔符
    delimiter = "```"
    client = OpenAiChat()

    # 第一步: 使用 OpenAI 的 Moderation API 检查用户输入是否合规或者是一个注入的 Prompt
    # output_moderation = client.moderation(user_input)
    # if output_moderation and output_moderation[0]["flagged"]:
    #     print("第一步：输入被 Moderation 拒绝")
    #     return "抱歉，您的请求不合规"
    # if debug: print("第一步：输入通过 Moderation 检查")

    # # 第二步：抽取出商品和对应的目录
    category_and_product_response = find_category_and_product(user_input, delimiter=delimiter)
    category_and_product_list = tools.read_string_to_object(category_and_product_response)
    if debug:
        print(f"第二步：抽取出商品列表")
        info = category_and_product_response.replace('\n', '').replace(' ', '')
        print(f"  ==> {info}")

    # 第三步：查找商品对应信息
    product_information = tools.generate_output_string(category_and_product_list)
    if debug:
        print("第三步：查找抽取出的商品信息")
        info = product_information.replace('\n', '').replace(' ', '')[:200]
        print(f"  ==> {info}")

    # 第四步：根据信息生成回答
    messages, response = assistant(user_input, product_information)
    # 将该轮信息加入到历史信息中
    context.extend(messages[1:])
    if debug:
        info = response.replace('\n', '').replace(' ', '')
        print("第四步：生成用户回答")
        print(f"  ==> {info}")

    # # 第五步：基于 Moderation API 检查输出是否合规
    # # response = openai.Moderation.create(input=final_response)
    # # moderation_output = response["results"][0]
    # # # 输出不合规
    # # if moderation_output["flagged"]:
    # #     if debug: print("第五步：输出被 Moderation 拒绝")
    # #     return "抱歉，我们不能提供该信息"
    # # if debug: print("第五步：输出经过 Moderation 检查")
    #
    # 第六步：模型检查是否很好地回答了用户问题
    evaluation_response = evaluation(user_input, response)
    evaluation_object = tools.read_string_to_object(evaluation_response)
    # print(evaluation_response)
    if debug:
        info = evaluation_response.replace('\n', '').replace(' ', '')
        print("第六步：模型评估该回答")
        print(f"  ==> {info}")

    # 第七步：如果评估为 Y，输出回答；如果评估为 N，反馈将由人工修正答案
    flag = evaluation_object.get("flag")
    if bool(flag):  # 使用 in 来避免模型可能生成 Yes
        if debug: print("第七步：模型赞同了该回答.")
    else:
        if debug: print("第七步：模型不赞成该回答.")
        response = "很抱歉，我无法提供您所需的信息。我将为您转接到一位人工客服代表以获取进一步帮助。"

    return response, context


if __name__ == "__main__":
    _user_input = "请介绍一下 SmartX ProPhone 智能手机和 FotoSnap 相机，包括单反相机。另外，介绍关于电视产品的信息。"
    # _response = find_category_and_product(_user_input)
    _response, _ = process_message(_user_input, [])
    print(_response)
