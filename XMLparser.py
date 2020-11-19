from bs4 import BeautifulSoup
from xml import etree
import pandas as pd
from os import listdir
from os.path import splitext
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

    pos[row.attrib['Id']] = len(dic['n_links'])  # position of the processed question in dic's values

    html = BeautifulSoup(row.attrib['Body']).find_all(['body', 'a', 'ul', 'ol'])

    dic['id'].append(int(row.attrib['Id']))
    dic['title'].append(BeautifulSoup(row.attrib['Title']).find('body').get_text())  # parse HTML in the title and append to dic
    dic['n_tags'].append(row.attrib['Tags'].count('<'))
    dic['time'].append(datetime.datetime.strptime(row.attrib['CreationDate'], '%Y-%m-%dT%H:%M:%S.%f'))
    dic['n_answers'].append(int(row.attrib['AnswerCount']))
    dic['score'].append(int(row.attrib['Score']))

    dic['time_til_first_answer'].append(float('inf'))  # this will be changed if an answer to this question is processed later

    dic['n_links'].append(0)
    dic['n_lists'].append(0)

    for t in html:
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

    posParent = pos[row.attrib['ParentId']]  # position of the answer's question in dic's values

    if dic['time_til_first_answer'][posParent] == float('inf'):
        time = datetime.datetime.strptime(row.attrib['CreationDate'], '%Y-%m-%dT%H:%M:%S.%f')
        dic['time_til_first_answer'][posParent] = (time - dic['time'][posParent])/datetime.timedelta(hours=1)


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

    dic = {'folder': [], 'id': [], 'body': [], 'title': [], 'n_links': [], 'n_tags': [],
           'n_lists': [], 'time': [], 'n_answers': [], 'score': [], 'time_til_first_answer': []}
    pos = {}  # dictionary (key = Id, value = position) to keep track of the position in dic's values where each question Id lies

    for file in listdir(folder):

        if splitext(file)[1] == '.xml':  # check for the right extension

            xmlroot = etree.ElementTree.ElementTree(file=file).getroot()

            for row in xmlroot:

                if row.attrib['PostTypeId'] == '1':
                    dic['folder'].append(folder)
                    processQuestion(row, dic, pos)

                if row.attrib['PostTypeId'] == '2':
                    processAnswer(row, dic, pos)

    df = pd.DataFrame(dic)
    df['y'] = df['score'] * df['n_answers'] / (df['time_til_first_answer'] + 1e-8)
    
    return df


if __name__ == '__main__':
    XMLparser('./')