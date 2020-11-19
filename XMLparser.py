from bs4 import BeautifulSoup
from xml.etree.ElementTree import ElementTree
import numpy as np
import pandas as pd
from os import listdir
from os.path import splitext, join
import datetime


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
    pos[row.attrib['Id']] = len(dic['n_links'])

    soup = BeautifulSoup(row.attrib['Body'], 'html.parser')
    html_tags = soup.find_all(['a', 'ul', 'ol'])

    dic['id'].append(int(row.attrib['Id']))
    dic['body'].append(soup.get_text())
    dic['title'].append(row.attrib['Title'])
    dic['n_tags'].append(row.attrib['Tags'].count('<'))
    dic['n_views'].append(int(row.attrib['ViewCount']))
    dic['time'].append(
        datetime.datetime.strptime(
            row.attrib['CreationDate'], '%Y-%m-%dT%H:%M:%S.%f'))
    dic['n_answers'].append(int(row.attrib['AnswerCount']))
    dic['score'].append(int(row.attrib['Score']))

    # This will be changed if an answer to this question is processed later:
    dic['time_til_first_answer'].append(float('inf'))

    dic['n_links'].append(0)
    dic['n_lists'].append(0)

    for t in html_tags:
        if t.name == 'a':
            dic['n_links'][-1] += 1
        elif t.name == 'ul' or t.name == 'ol':
            dic['n_lists'][-1] += 1
        else:
            dic['body'].append(t.get_text())


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


def XMLparser(folder):
    '''
    Function that parses 'Posts' XML files from the Stack Exchange dataset
    to extract the body of the question, the numer of links, the numer of
    tags, and the number of lists.

    Inputs:
        folder (str): path to the XML file to past

    Outputs:
        df (pandas dataframe): dataframe with columns 'body', 'title',
                                'n_links', 'n_tags', 'n_lists', and 'y'
    '''

    dic = {'folder': [], 'id': [], 'body': [], 'title': [], 'n_links': [],
           'n_tags': [], 'n_lists': [], 'n_views': [], 'time': [],
           'n_answers': [], 'score': [], 'time_til_first_answer': []}

    # Dictionary (key = Id, value = position) to keep track of the position in
    # dic's values where each question Id lies:
    pos = {}

    for file in listdir(folder):

        if splitext(file)[1] == '.xml':  # check for the right extension

            xmlroot = ElementTree(file=join(folder, file)).getroot()

            for row in xmlroot:

                if row.attrib['PostTypeId'] == '1':
                    dic['folder'].append(folder)
                    processQuestion(row, dic, pos)

                if row.attrib['PostTypeId'] == '2':
                    processAnswer(row, dic, pos)

    df = pd.DataFrame(dic)
    df['y'] = df['score'].map(lambda x: x + 1 if x >= 0 else np.exp(x)) \
        * df['n_answers'] / df['time_til_first_answer']

    return df
