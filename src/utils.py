import dgl
import pandas as pd
import os
import torch
import logging

def collate_data(data):
    
    subgraph_list1, subgraph_list2, subgraph_list3, label_list = map(list, zip(*data))
    subgraph1 = dgl.batch(subgraph_list1)
    subgraph2 = dgl.batch(subgraph_list2)
    subgraph3 = dgl.batch(subgraph_list3)
    g_label = torch.stack(label_list)
    
    return subgraph1, subgraph2, subgraph3, g_label
    
def get_logger(name, path):
    
    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=path)
    
    console.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

def kddcup_load():
    path = 'data/kddcup15'

    course_date = pd.read_csv(os.path.join(path, 'date.csv'))
    course_info = pd.read_csv(os.path.join(path, 'object.csv'))
    enrollment_train = pd.read_csv(os.path.join(path, 'enrollment_train.csv'))
    enrollment_test = pd.read_csv(os.path.join(path, 'enrollment_test.csv'))
    log_train = pd.read_csv(os.path.join(path, 'log_train.csv'))
    log_test = pd.read_csv(os.path.join(path, 'log_test.csv'))
    truth_train = pd.read_csv(os.path.join(path, 'truth_train.csv'), header=None, names=['enrollment_id', 'truth'])
    truth_test = pd.read_csv(os.path.join(path, 'truth_test.csv'), header=None, names=['enrollment_id', 'truth'])
    
    truth_all = pd.concat([truth_train, truth_test])
    log_all = pd.concat([log_train, log_test])
    enrollment_all = pd.concat([enrollment_train, enrollment_test])

    return course_date, course_info, enrollment_train, enrollment_test, log_train, log_test, truth_train, truth_test, truth_all, log_all, enrollment_all

def naver_load():
    path = 'data/naver'
    
    cs = pd.read_excel(os.path.join(path, '01.CS_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    web = pd.read_excel(os.path.join(path, '02.개발-웹_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23-1.video_play_log', '23-2.video_play_log'])
    app = pd.read_excel(os.path.join(path, '03.개발-앱_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    general = pd.read_excel(os.path.join(path, '04.개발-일반_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    design = pd.read_excel(os.path.join(path, '05.디자인-웹_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    ai = pd.read_excel(os.path.join(path, '06.AI_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    ds = pd.read_excel(os.path.join(path, '07.DS_20220516.xlsx'), sheet_name=['02.chapter', '03.comment', '05.course', '08.evaluation_history', '09.forum', '10.lecture', '18.sec_user_course_role', '22.vote', '23.video_play_log'])
    
    course_info=pd.concat([cs['05.course'], ai['05.course'], ds['05.course'], web['05.course'], app['05.course'], general['05.course'], design['05.course']]).reset_index(drop=True)
    chapter_list = pd.concat([cs['02.chapter'], ai['02.chapter'], ds['02.chapter'], web['02.chapter'], app['02.chapter'], general['02.chapter'], design['02.chapter']]).reset_index(drop=True) # 
    lecture_list = pd.concat([cs['10.lecture'], ai['10.lecture'], ds['10.lecture'], web['10.lecture'], app['10.lecture'], general['10.lecture'], design['10.lecture']]).reset_index(drop=True) # lecture, quiz

    evaluation_log = pd.concat([cs['08.evaluation_history'], ai['08.evaluation_history'], ds['08.evaluation_history'], web['08.evaluation_history'], app['08.evaluation_history'], general['08.evaluation_history'], design['08.evaluation_history']]).reset_index(drop=True) # problem
    forum_log = pd.concat([cs['09.forum'], ai['09.forum'], ds['09.forum'], web['09.forum'], app['09.forum'], general['09.forum'], design['09.forum']]).reset_index(drop=True) # discussion
    video_play_log = pd.concat([cs['23.video_play_log'], ai['23.video_play_log'], ds['23.video_play_log'], web['23-1.video_play_log'],  web['23-2.video_play_log'], app['23.video_play_log'], general['23.video_play_log'], design['23.video_play_log']]).reset_index(drop=True) # video
    comment_log = pd.concat([cs['03.comment'], ai['03.comment'], ds['03.comment'], web['03.comment'], app['03.comment'], general['03.comment'], design['03.comment']]).reset_index(drop=True)

    user_info = pd.concat([cs['18.sec_user_course_role'], ai['18.sec_user_course_role'], ds['18.sec_user_course_role'], web['18.sec_user_course_role'], app['18.sec_user_course_role'], general['18.sec_user_course_role'], design['18.sec_user_course_role']]).reset_index(drop=True)

    return course_info, chapter_list, lecture_list, evaluation_log, forum_log, video_play_log, comment_log, user_info