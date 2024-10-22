import os
import datetime
import numpy as np
import pandas as pd
from xml.etree.ElementTree import ElementTree


def processQuestion(row, dic, pos):
    '''
    Function that processes questions in 'Posts' XML files from the Stack
    Exchange dataset

    Inputs:
        row (XML Element): row of the XML file being processed
        dic (dict): dictionary where information of interest is stored
        pos (dict): dictionary that keeps the positions of questions in
                    dict's values

    Outputs: None (it modifies pos and dic in place)
    '''

    # Position of the processed question in dic's values:
    pos[row.attrib['Id']] = len(dic['id'])

    dic['id'].append(int(row.attrib['Id']))
    dic["body"].append(row.attrib['Body'])
    dic['title'].append(row.attrib['Title'])
    dic['tags'].append(row.attrib['Tags'])
    dic['n_views'].append(int(row.attrib['ViewCount']))
    dic['time'].append(
        datetime.datetime.strptime(
            row.attrib['CreationDate'], '%Y-%m-%dT%H:%M:%S.%f'))
    dic['n_answers'].append(int(row.attrib['AnswerCount']))
    dic['score'].append(int(row.attrib['Score']))

    # This will be changed if an answer to this question is processed later:
    dic['time_til_first_answer'].append(float('inf'))


def processAnswer(row, dic, pos):
    '''
    Function that processes answers in 'Posts' XML files from the Stack
    Exchange dataset

    Inputs:
        row (XML Element): row of the XML file being processed
        dic (dict): dictionary where information of interest is stored
        pos (dict): dictionary that keeps the positions of questions in
                    dict's values

    Outputs: None (it modifies dic in place)
    '''
    try:
        # Position of the answer's question in dic's values:
        posParent = pos[row.attrib['ParentId']]

        if dic['time_til_first_answer'][posParent] == float('inf'):
            time = datetime.datetime.strptime(
                row.attrib['CreationDate'], '%Y-%m-%dT%H:%M:%S.%f')
            dic['time_til_first_answer'][posParent] = max(
                1/3600,
                (time - dic['time'][posParent])/datetime.timedelta(hours=1))
    except KeyError:
        print("[ERROR] Parent not found: ", row.attrib['ParentId'])


def XMLparser(path_str: str):
    '''
    Function that parses 'Posts' XML files from the Stack Exchange dataset
    to extract the body of the question, the numer of links, the numer of
    tags, and the number of lists.

    Inputs:
        path_str (str): path to the XML Posts file or a dir that contains one

    Outputs:
        df (pandas dataframe): dataframe with columns 'body', 'title',
                                'n_links', 'n_tags', 'n_lists', and 'y'
    '''

    dic = {'folder': [], 'id': [], 'body': [], 'title': [],
           'tags': [], 'n_views': [], 'time': [],
           'n_answers': [], 'score': [], 'time_til_first_answer': []}

    # Dictionary (key = Id, value = position) to keep track of the position in
    # dic's values where each question Id lies:
    pos = {}

    # Get path to Posts.xml file
    if os.path.isdir(path_str):
        if "Posts.xml" not in os.listdir(path_str):
            raise FileNotFoundError("Posts.xml file not found in dir!")
        filepath = os.path.join(path_str, "Posts.xml")
    elif os.path.isfile(path_str) and path_str.endswith('.xml'):
        filepath = path_str
    else:
        raise FileNotFoundError("Invalid path!")

    xmlroot = ElementTree(file=filepath).getroot()
    for row in xmlroot:

        if row.attrib['PostTypeId'] == '1':
            dic['folder'].append(path_str)
            processQuestion(row, dic, pos)

        if row.attrib['PostTypeId'] == '2':
            processAnswer(row, dic, pos)

    # for file in os.listdir(path_str):
    #
    #     if os.path.splitext(file)[1] == '.xml':  # check for the right extension
    #
    #         xmlroot = ElementTree(file=os.path.join(path_str, file)).getroot()
    #
    #         for row in xmlroot:
    #
    #             if row.attrib['PostTypeId'] == '1':
    #                 dic['folder'].append(path_str)
    #                 processQuestion(row, dic, pos)
    #
    #             if row.attrib['PostTypeId'] == '2':
    #                 processAnswer(row, dic, pos)

    df = pd.DataFrame(dic)
    df['y'] = df['score'].map(lambda x: x + 1 if x >= 0 else np.exp(x)) \
        * df['n_answers'] / df['time_til_first_answer']

    return df
