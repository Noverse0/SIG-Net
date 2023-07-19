import pandas as pd
import numpy as np
import dgl
from dgl.data import DGLDataset
import torch
from tqdm import tqdm
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import kddcup_load, naver_load

warnings.filterwarnings(action='ignore')

def get_subgraph(graph:dgl.graph,
                        o_with_target:torch.tensor,
                        e_with_student:torch.tensor, 
                        target_course:torch.tensor, 
                        target_enroll:torch.tensor,
                        ):
    nodes = torch.unique(torch.cat([o_with_target, e_with_student, target_course, target_enroll], dim=0,))
    
    nodes = nodes.type(torch.int64)
    subgraph = dgl.node_subgraph(graph, nodes, store_ids=True) 
    
    return subgraph

class kddcup_graph(DGLDataset):
    def __init__(self):
        super().__init__(name='kddcup')
        
    def process(self):
        course_date, course_info, enrollment_train, enrollment_test, log_train, log_test, truth_train, truth_test, truth_all, log_all, enrollment_all = kddcup_load()
        
        # days 계산
        log_all['date'] = log_all['time'].str.split('T').str[0]
        log_all = pd.merge(log_all, enrollment_all, how='left')
        log_all=pd.merge(log_all, course_date, how='left', on='course_id')
        delta_days=pd.to_datetime(log_all['date'], format='%Y-%m-%d', errors='raise') - pd.to_datetime(log_all['from'], format='%Y-%m-%d', errors='raise')
        log_all['days'] = delta_days.dt.days

        truth_train = pd.merge(truth_train, enrollment_train, how='left', on='enrollment_id')
        truth_test = pd.merge(truth_test, enrollment_test, how='left', on='enrollment_id')
        truth_all = pd.merge(truth_all, enrollment_all, how='left', on='enrollment_id')

        log_all = log_all.replace({'event' : 'problem'}, 0)
        log_all = log_all.replace({'event' : 'video'}, 1)
        log_all = log_all.replace({'event' : 'discussion'}, 2)
        log_all = log_all.replace({'event' : 'wiki'}, 3)
        log_all = log_all.replace({'event' : 'page_close'}, 3)
        log_all = log_all.replace({'event' : 'navigate'}, 3)
        log_all = log_all.replace({'event' : 'access'}, 3)

        all_clicksum = log_all[['enrollment_id', 'days', 'object', 'event', 'time']].groupby(['enrollment_id', 'event', 'days', 'object']).count().reset_index()
        all_clicksum.rename(columns = {'time':'click_sum'}, inplace=True)

        # Split 10 days
        one_day = list(range(0,10))
        two_day = list(range(10,20))
        third_day = list(range(20,30))

        one_log_all = log_all[log_all['days'].isin(one_day)].copy()
        two_log_all = log_all[log_all['days'].isin(two_day)].copy()
        third_log_all = log_all[log_all['days'].isin(third_day)].copy()

        # problem: 0, video: 1, discussion: 2, etc: 3
        activity = [0, 1, 2, 3]
        activity_log_all = log_all[log_all['event'].isin(activity)]
        one_activity_log_all = one_log_all[one_log_all['event'].isin(activity)]
        two_activity_log_all = two_log_all[two_log_all['event'].isin(activity)]
        third_activity_log_all = third_log_all[third_log_all['event'].isin(activity)]

        one_activity_log_all=pd.merge(one_activity_log_all, one_activity_log_all.groupby(['enrollment_id', 'object']).count().reset_index()[['enrollment_id', 'object', 'time']].rename(columns={'time':'clicksum'}), how='left')
        two_activity_log_all=pd.merge(two_activity_log_all, two_activity_log_all.groupby(['enrollment_id', 'object']).count().reset_index()[['enrollment_id', 'object', 'time']].rename(columns={'time':'clicksum'}), how='left')
        third_activity_log_all=pd.merge(third_activity_log_all, third_activity_log_all.groupby(['enrollment_id', 'object']).count().reset_index()[['enrollment_id', 'object', 'time']].rename(columns={'time':'clicksum'}), how='left')

        enroll_list = list(set(truth_all['enrollment_id'])) # enrollemnt set
        object_list = list(set(activity_log_all['object'])) # object set
        course_list = list(set(course_date['course_id'])) # course set

        # make node id
        enroll_node_id, object_node_id, course_node_id, enroll_truth = {}, {}, {}, {}
        index = 0
        for i in enroll_list:
            enroll_node_id[i] = index
            enroll_truth[i] = truth_all[truth_all['enrollment_id']==i]['truth'].values[0]
            index += 1
        for i in object_list:
            object_node_id[i] = index
            index += 1
        for i in course_list:
            course_node_id[i] = index
            index += 1

        enroll_num_node = len(enroll_node_id)
        object_num_node = len(object_node_id)
        course_num_node = len(course_node_id)
        num_node = enroll_num_node + object_num_node + course_num_node

        one_enroll_interaction = one_activity_log_all[['enrollment_id', 'event', 'object', 'clicksum']].groupby(['enrollment_id', 'event', 'object', 'clicksum']).nunique().reset_index()
        two_enroll_interaction = two_activity_log_all[['enrollment_id', 'event', 'object', 'clicksum']].groupby(['enrollment_id', 'event', 'object', 'clicksum']).nunique().reset_index()
        third_enroll_interaction = third_activity_log_all[['enrollment_id', 'event', 'object', 'clicksum']].groupby(['enrollment_id', 'event', 'object', 'clicksum']).nunique().reset_index()

        object_info = pd.merge(activity_log_all, truth_all, how='left')
        object_info = object_info[['course_id', 'object', 'event']].groupby(['course_id', 'object', 'event']).nunique().reset_index()

        print('feature extraction')
        enroll_feature = defaultdict(list)
        labels1, labels2, labels3, labels0 = [], [], [], []
        
        for enroll_id in tqdm(enroll_list):
            feature = np.zeros(30)
            for index, datarow in all_clicksum[all_clicksum['enrollment_id']==enroll_id].iterrows():
                feature[datarow['days']] = datarow['click_sum']
            enroll_feature[enroll_id] = feature
            if sum(feature[:10]) == 0:
                labels0.append(1)
            else:
                labels0.append(0)
            if sum(feature[10:20]) == 0:
                labels1.append(1)
            else:
                labels1.append(0)
            if sum(feature[20:30]) == 0:
                labels2.append(1)
            else:
                labels2.append(0)
            labels3.append(enroll_truth[enroll_id])
            
        labels_df = pd.DataFrame(zip(enroll_list, labels0, labels1, labels2, labels3))
        labels_df.columns = ['enroll_id', 'labels0', 'labels1', 'labels2', 'labels3']
            
        object_feature = dict()
        object_event = activity_log_all[['event', 'object']].groupby(['event', 'object']).nunique().reset_index()
        for index, datarow in tqdm(object_event.iterrows(), total=object_event.shape[0]):
            feature = np.zeros(len(activity))
            feature[datarow['event']] = 1
            object_feature[object_node_id[datarow['object']]] = feature
            
        e_in_cousre_dic = defaultdict(list)
        course_feature = dict()
        course_cate_list = ['video', 'problem', 'discussion', 'chapter', 'html', 'about']
        course_cate_to_index = {category : index for index, category in enumerate(course_cate_list)}
        coures_cate_df = course_info[['course_id', 'category', 'module_id']].groupby(['course_id', 'category']).count().reset_index()
        coures_cate_df = coures_cate_df[coures_cate_df['category'].isin(course_cate_list)]
        coures_cate_df.rename(columns = {'module_id':'count'}, inplace=True)

        for course in course_list:
            for enroll in truth_all[truth_all['course_id'] == course]['enrollment_id']:
                e_in_cousre_dic[course_node_id[course]].append(enroll_node_id[enroll])
            feature = np.zeros(len(course_cate_list)+1)
            for index, datarow in coures_cate_df[coures_cate_df['course_id'] == course].iterrows():
                feature[course_cate_to_index[datarow['category']]] = datarow['count']
            feature[-1] = len(e_in_cousre_dic[course_node_id[course]])
            course_feature[course_node_id[course]] = feature
            
        stan_scaler = StandardScaler()

        col_name = []
        for i in range(30):
            col_name.append(f'clicksum_day{i}')
            
        enroll_feature_df = pd.DataFrame(enroll_feature, index=col_name).T
        enroll_feature_df = enroll_feature_df.reset_index().rename(columns={'index': 'enrollment_id'})
        enroll_feature_df = pd.merge(enroll_feature_df, enrollment_all[['enrollment_id', 'course_id']], how='left', on='enrollment_id').set_index('enrollment_id', drop=True)

        node_feature_df = pd.DataFrame()
        for course_id in course_node_id.keys():
            stan_df = enroll_feature_df[enroll_feature_df['course_id'] == course_id].drop('course_id', 1)
            index_name = stan_df.index.tolist()
            node_feature_df = pd.concat([node_feature_df, pd.DataFrame(stan_scaler.fit_transform(stan_df), index = index_name, columns = col_name)])
            
        course_feature_df = pd.DataFrame(course_feature, index=course_cate_list+['total_enrollment']).T
        index_name = course_feature_df.index.tolist()
        course_feature_df = pd.DataFrame(stan_scaler.fit_transform(course_feature_df), index = index_name, columns = course_cate_list+['total_enrollment'])
        
        #-------------------------------make Graph------------------------------
        print('Graph building')
        
        src_node, dst_node = [], []
        one_etype, two_etype, third_etype = [], [], []
        one_node_feature, two_node_feature, third_node_feature = [], [], []
        non_feature = []
        o_in_cousre_dic = defaultdict(list) 
        one_e_in_object_dic, one_o_in_enroll_dic, one_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        two_e_in_object_dic, two_o_in_enroll_dic, two_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        third_e_in_object_dic, third_o_in_enroll_dic, third_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        e_with_student_dic = dict()

        # Make Object-Course Edge
        for index, data_row in tqdm(object_info.iterrows(), total=object_info.shape[0]):
            src_node.append(object_node_id[data_row['object']])
            dst_node.append(course_node_id[data_row['course_id']])
            
            if data_row['event'] == 0: # problem
                one_etype.append(0)
                two_etype.append(0)
                third_etype.append(0)
            if data_row['event'] == 1: # video
                one_etype.append(1)
                two_etype.append(1)
                third_etype.append(1)
            if data_row['event'] == 2: # discusstion
                one_etype.append(2)
                two_etype.append(2)
                third_etype.append(2)
            if data_row['event'] == 3: # wiki
                one_etype.append(3)
                two_etype.append(3)
                third_etype.append(3)
            
            o_in_cousre_dic[course_node_id[data_row['course_id']]].append(object_node_id[data_row['object']])
                
        # Make Enrollemt-Object Edge
        one_src_node, one_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(one_enroll_interaction.iterrows(), total = one_enroll_interaction.shape[0]):
            one_src_node.append(enroll_node_id[data_row['enrollment_id']])
            one_dst_node.append(object_node_id[data_row['object']])
            
            if data_row['event'] == 0: # problem
                one_etype.append(4)
            if data_row['event'] == 1: # video
                one_etype.append(5)
            if data_row['event'] == 2: # discusstion
                one_etype.append(6)
            if data_row['event'] == 3: # wiki
                one_etype.append(7)
                
            one_e_in_object_dic[object_node_id[data_row['object']]].append(enroll_node_id[data_row['enrollment_id']])
            one_o_in_enroll_dic[enroll_node_id[data_row['enrollment_id']]].append(object_node_id[data_row['object']])
            
        two_src_node, two_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(two_enroll_interaction.iterrows(), total = two_enroll_interaction.shape[0]):
            two_src_node.append(enroll_node_id[data_row['enrollment_id']])
            two_dst_node.append(object_node_id[data_row['object']])

            if data_row['event'] == 0: # problem
                two_etype.append(4)
            if data_row['event'] == 1: # video
                two_etype.append(5)
            if data_row['event'] == 2: # discusstion
                two_etype.append(6)
            if data_row['event'] == 3: # wiki
                two_etype.append(7)
                
            two_e_in_object_dic[object_node_id[data_row['object']]].append(enroll_node_id[data_row['enrollment_id']])
            two_o_in_enroll_dic[enroll_node_id[data_row['enrollment_id']]].append(object_node_id[data_row['object']])
            
        third_src_node, third_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(third_enroll_interaction.iterrows(), total = third_enroll_interaction.shape[0]):
            third_src_node.append(enroll_node_id[data_row['enrollment_id']])
            third_dst_node.append(object_node_id[data_row['object']])

            if data_row['event'] == 0: # problem
                third_etype.append(4)
            if data_row['event'] == 1: # video
                third_etype.append(5)
            if data_row['event'] == 2: # discusstion
                third_etype.append(6)
            if data_row['event'] == 3: # wiki
                third_etype.append(7)
                
            third_e_in_object_dic[object_node_id[data_row['object']]].append(enroll_node_id[data_row['enrollment_id']])
            third_o_in_enroll_dic[enroll_node_id[data_row['enrollment_id']]].append(object_node_id[data_row['object']])
            
        one_src_node, one_dst_node = one_src_node + one_dst_node, one_dst_node + one_src_node
        two_src_node, two_dst_node = two_src_node + two_dst_node, two_dst_node + two_src_node
        third_src_node, third_dst_node = third_src_node + third_dst_node, third_dst_node + third_src_node
        one_etype = one_etype + one_etype
        two_etype = two_etype + two_etype
        third_etype = third_etype + third_etype
        
        # Make Student Edge
        for student in tqdm(list(set(truth_all['username']))):
            student_list_df = truth_all[truth_all['username']==student]
            student_list = list(map((lambda x: enroll_node_id[x]), student_list_df['enrollment_id'].tolist()))
            
            student_truth1 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels1'].values[0]), student_list_df['enrollment_id'].tolist()))
            student_truth2 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels2'].values[0]), student_list_df['enrollment_id'].tolist()))
            student_truth3 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels3'].values[0]), student_list_df['enrollment_id'].tolist()))
            
            e_with_student_dic[student] = student_list
            
            n = len(student_list)
            for src_student in range(n):
                for dst_student in range(n):
                    if src_student != dst_student:
                        if student_truth1[src_student] == 1: # dropout
                            one_etype.append(8)
                        else: 
                            one_etype.append(9)
                        if student_truth2[src_student] == 1: # dropout
                            two_etype.append(8)
                        else: 
                            two_etype.append(9)
                        if student_truth3[src_student] == 1: # dropout
                            third_etype.append(8)
                        else: 
                            third_etype.append(9)
                        
                        one_student_eid_dic[student_list[src_student]].append(len(one_src_node))
                        two_student_eid_dic[student_list[src_student]].append(len(two_src_node))
                        third_student_eid_dic[student_list[src_student]].append(len(third_src_node))
                            
                        one_src_node.append(student_list[src_student])
                        one_dst_node.append(student_list[dst_student])
                        two_src_node.append(student_list[src_student])
                        two_dst_node.append(student_list[dst_student])
                        third_src_node.append(student_list[src_student])
                        third_dst_node.append(student_list[dst_student])
            
        one_graph = dgl.graph((one_src_node, one_dst_node), num_nodes = num_node)
        two_graph = dgl.graph((two_src_node, two_dst_node), num_nodes = num_node)
        third_graph = dgl.graph((third_src_node, third_dst_node), num_nodes = num_node)
        
        #-------------------------------ADD Node Feature------------------------------
        enroll_zero_list = np.zeros(10)
        cousre_zero_list = np.zeros(len(course_cate_list)+1)
        for e_node_id in tqdm(enroll_node_id.keys()):
            one_feature = np.array([1,0,0])
            two_feature = np.array([1,0,0])
            third_feature = np.array([1,0,0])
            
            one_feature = np.append(one_feature, node_feature_df.loc[e_node_id].tolist()[0:10])
            two_feature = np.append(two_feature, node_feature_df.loc[e_node_id].tolist()[10:20])
            third_feature = np.append(third_feature, node_feature_df.loc[e_node_id].tolist()[20:30])
            
            one_feature = np.append(one_feature, cousre_zero_list)
            two_feature = np.append(two_feature, cousre_zero_list)
            third_feature = np.append(third_feature, cousre_zero_list)
            
            one_node_feature.append(one_feature)
            two_node_feature.append(two_feature)
            third_node_feature.append(third_feature)
            
            non_feature.append([1,0,0])
            
        for o_node_id in tqdm(object_node_id.values()):
            feature = np.array([0,1,0])
            feature = np.append(feature, enroll_zero_list)
            feature = np.append(feature, cousre_zero_list)
            
            one_node_feature.append(feature)
            two_node_feature.append(feature)
            third_node_feature.append(feature)
            
            non_feature.append([0,1,0])
            
        for c_node_id in tqdm(course_node_id.values()):
            feature = np.array([0,0,1])
            feature = np.append(feature, enroll_zero_list)
            feature = np.append(feature, course_feature_df.loc[c_node_id].tolist())
            
            one_node_feature.append(feature)
            two_node_feature.append(feature)
            third_node_feature.append(feature)
            
            non_feature.append([0,0,1])
            
        one_graph.ndata['feature'] = torch.FloatTensor(one_node_feature)
        two_graph.ndata['feature'] = torch.FloatTensor(two_node_feature)
        third_graph.ndata['feature'] = torch.FloatTensor(third_node_feature)
        
        one_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)
        two_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)
        third_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)
        
        one_graph.edata['etype'] = torch.tensor(one_etype)
        two_graph.edata['etype'] = torch.tensor(two_etype)
        third_graph.edata['etype'] = torch.tensor(third_etype)
        
        self.one_graph = one_graph
        self.two_graph = two_graph
        self.third_graph = third_graph
        self.truth_all = truth_all
        self.labels_df = labels_df
        self.enroll_node_id = enroll_node_id
        self.object_node_id = object_node_id
        self.course_node_id = course_node_id
        self.e_in_cousre_dic = e_in_cousre_dic
        self.o_in_cousre_dic = o_in_cousre_dic
        self.e_with_student_dic = e_with_student_dic
        
        self.one_student_eid_dic = one_student_eid_dic
        self.two_student_eid_dic = two_student_eid_dic
        self.third_student_eid_dic = third_student_eid_dic
        
        self.one_o_in_enroll_dic = one_o_in_enroll_dic
        self.two_o_in_enroll_dic = two_o_in_enroll_dic
        self.third_o_in_enroll_dic = third_o_in_enroll_dic
        self.one_e_in_object_dic = one_e_in_object_dic
        self.two_e_in_object_dic = two_e_in_object_dic
        self.third_e_in_object_dic = third_e_in_object_dic
        
    def __getitem__(self, i):
        if i == 0:
            return self.one_graph
            
        target_enroll = self.enroll_node_id[i]
        enroll_info = self.truth_all[self.truth_all['enrollment_id'] == i]
        target_course = self.course_node_id[enroll_info['course_id'].values[0]]
        student_name = enroll_info['username'].values[0]
        
        o_with_target = self.o_in_cousre_dic[target_course]
        enroll_list = self.e_with_student_dic[student_name]
        
        one_subgraph = get_subgraph(graph = self.one_graph,
                                    o_with_target=torch.tensor(o_with_target), 
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        two_subgraph = get_subgraph(graph = self.two_graph,
                                    o_with_target=torch.tensor(o_with_target), 
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        third_subgraph = get_subgraph(graph = self.third_graph,
                                    o_with_target=torch.tensor(o_with_target), 
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        target_mask = np.zeros(one_subgraph.num_nodes())
        target_enroll_id = (one_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (one_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        one_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        target_mask = np.zeros(two_subgraph.num_nodes())
        target_enroll_id = (two_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (two_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        two_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        target_mask = np.zeros(third_subgraph.num_nodes())
        target_enroll_id = (third_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (third_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        third_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        # Masking student edge
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.one_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (one_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: one_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((one_subgraph.edges()[0] == e_id) & (one_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        one_subgraph.edata['etype'][drop_etype_tensor.tolist()] = one_subgraph.edata['etype'][new_etype]
        
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.two_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (two_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: two_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((two_subgraph.edges()[0] == e_id) & (two_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        two_subgraph.edata['etype'][drop_etype_tensor.tolist()] = two_subgraph.edata['etype'][new_etype]
        
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.third_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (third_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: third_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((third_subgraph.edges()[0] == e_id) & (third_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        third_subgraph.edata['etype'][drop_etype_tensor.tolist()] = third_subgraph.edata['etype'][new_etype]
        
        return one_subgraph, two_subgraph, third_subgraph, torch.Tensor(self.labels_df[self.labels_df['enroll_id']==i].values[-1][1:])
        

class naver_graph(DGLDataset):
    def __init__(self):
        super().__init__(name='naver')
        
    def process(self):
        course_info, chapter_list, lecture_list, evaluation_log, forum_log, video_play_log, comment_log, user_info = naver_load()
        
        course_list = course_info['id'].tolist()
        course_list.remove(12134) # eliminate

        video_play_log = video_play_log[video_play_log['course_id'].isin(course_list)]
        evaluation_log = evaluation_log[evaluation_log['course_id'].isin(course_list)]
        forum_log = forum_log[forum_log['course_id'].isin(course_list)]
        comment_log = comment_log[comment_log['course_id'].isin(course_list)]

        lecture_list = lecture_list[lecture_list['course_id'].isin(course_list)]
        object_list = lecture_list['id'].tolist() + list(set(forum_log['bbs_id']))

        user_info = user_info[user_info['role_priority'] == 2]
        user_info = user_info[user_info['course_id'].isin(course_list)]
        enroll_list = user_info['id'].tolist()

        video_play_log = video_play_log[['course_id', 'reg_ymdt', 'user_id', 'video_type', 'video_type_id']].rename(columns={'reg_ymdt':'ymdt', 
                                                                                                                            'sec_user_id':'user_id',
                                                                                                                            'video_type':'type',
                                                                                                                            'video_type_id':'type_id',
                                                                                                                            })
        evaluation_log = evaluation_log[evaluation_log['evaluation_history_type'].isin(['QUIZ'])]
        evaluation_log = evaluation_log[['course_id', 'submit_ymdt', 'sec_user_id', 'evaluation_history_type', 'evaluation_history_type_id']].rename(columns={'submit_ymdt':'ymdt', 
                                                                                                                                                            'sec_user_id':'user_id',
                                                                                                                                                            'evaluation_history_type':'type',
                                                                                                                                                            'evaluation_history_type_id':'type_id',
                                                                                                                                                            })
        forum_log = forum_log[['course_id', 'reg_ymdt', 'reg_id', 'bbs_id']].rename(columns={'reg_ymdt':'ymdt', 
                                                                                            'bbs_id':'type_id',
                                                                                            'reg_id':'user_id'})
        forum_log['type'] = 'FORUM'
        comment_log = comment_log[['course_id', 'reg_ymdt', 'reg_id', 'bbs_id']].rename(columns={'reg_ymdt':'ymdt', 
                                                                                            'bbs_id':'type_id',
                                                                                            'reg_id':'user_id'})
        comment_log['type'] = 'COMMENT'

        log_all = pd.concat([video_play_log, evaluation_log, forum_log, comment_log])
        
        log_all=log_all.dropna()
        log_all = pd.merge(log_all, user_info[['id', 'course_id', 'sec_user_id']].rename(columns={'sec_user_id':'user_id', 'id':'enroll_id'}), how='left', on=['user_id', 'course_id'])
        log_all['date'] = log_all['ymdt'].str.split(' ').str[0]
        log_all=pd.merge(log_all, log_all.sort_values('date').groupby('enroll_id').first().reset_index()[['enroll_id', 'date']].rename(columns={'date':'start_date'}), how='left', on='enroll_id')
        delta_days=pd.to_datetime(log_all['date'], format='%Y-%m-%d', errors='raise') - pd.to_datetime(log_all['start_date'], format='%Y-%m-%d', errors='raise')
        log_all['days'] = delta_days.dt.days
        days = range(40)
        log_all=log_all[log_all['days'].isin(days)]
        all_clicksum = log_all.groupby(['enroll_id', 'days']).count().reset_index()[['enroll_id', 'days', 'date']]
        all_clicksum.rename(columns = {'date':'click_sum'}, inplace=True)

        print('feature extraction')

        enroll_feature = defaultdict(list)
        enroll_id_list, labels1, labels2, labels3, labels0 = [], [], [], [], []
        for enroll_id in tqdm(enroll_list):
            feature = np.zeros(40)
            for index, datarow in all_clicksum[all_clicksum['enroll_id']==enroll_id].iterrows():
                feature[int(datarow['days'])] = datarow['click_sum']
            if sum(feature) != 0: # log기록이 하나도 없는 학생
                if sum(feature[30:]) == 0: # dropout = 1
                    labels3.append(1)
                else:
                    labels3.append(0)
                if sum(feature[20:30]) == 0: # dropout = 1
                    labels2.append(1)
                else:
                    labels2.append(0)
                if sum(feature[10:20]) == 0: # dropout = 1
                    labels1.append(1)
                else:
                    labels1.append(0)
                if sum(feature[:10]) == 0: # dropout = 1
                    labels0.append(1)
                else:
                    labels0.append(0)
                enroll_id_list.append(enroll_id)
                enroll_feature[enroll_id] = feature[:30]
                
        labels_df = pd.DataFrame(zip(enroll_id_list, labels0, labels1, labels2, labels3))
        labels_df.columns = ['enroll_id', 'labels0','labels1', 'labels2', 'labels3']

        log_all = log_all[log_all['enroll_id'].isin(enroll_feature.keys())]
        enroll_list = list(set(labels_df['enroll_id']))
        object_list = list(set(log_all['type_id']))
        course_list = list(set(log_all['course_id']))
        enroll_node_id, object_node_id, course_node_id = {}, {}, {}
        index = 0
        for i in enroll_list:
            enroll_node_id[i] = index
            index += 1
        for i in object_list:
            object_node_id[i] = index
            index += 1
        for i in course_list:
            course_node_id[i] = index
            index += 1

        enroll_num_node = len(enroll_node_id)
        object_num_node = len(object_node_id)
        course_num_node = len(course_node_id)
        num_node = enroll_num_node + object_num_node + course_num_node

        e_in_cousre_dic = defaultdict(list)
        course_feature = dict()
        course_cate_list = ['LECTURE', 'QUIZ', 'COMMENT', 'FORUM', 'INFOPAGE', 'INTRODUCTION']
        course_cate_to_index = {category : index for index, category in enumerate(course_cate_list)}
        coures_cate_df = log_all[['course_id', 'type', 'type_id']].groupby(['course_id', 'type', 'type_id']).first().reset_index().groupby(['course_id', 'type']).count().reset_index()
        coures_cate_df = coures_cate_df[coures_cate_df['type'].isin(course_cate_list)]
        coures_cate_df.rename(columns = {'type_id':'count'}, inplace=True)

        for course in course_list:
            feature = np.zeros(len(course_cate_list))
            for index, datarow in coures_cate_df[coures_cate_df['course_id'] == course].iterrows():
                feature[course_cate_to_index[datarow['type']]] = datarow['count']
            course_feature[course] = feature
            
        # LECTURE: 0, QUIZ: 1, DISCUSSION: 2, ETC: 3
        log_all = log_all.replace({'type' : 'LECTURE'}, 0)
        log_all = log_all.replace({'type' : 'QUIZ'}, 1)
        log_all = log_all.replace({'type' : 'COMMENT'}, 2)
        log_all = log_all.replace({'type' : 'FORUM'}, 2)
        log_all = log_all.replace({'type' : 'INFOPAGE'}, 3)
        log_all = log_all.replace({'type' : 'INTRODUCTION'}, 3)

        # Split 10 days
        one_day = list(range(0,10))
        two_day = list(range(10,20))
        third_day = list(range(20,30))

        one_log_all = log_all[log_all['days'].isin(one_day)].copy()
        two_log_all = log_all[log_all['days'].isin(two_day)].copy()
        third_log_all = log_all[log_all['days'].isin(third_day)].copy()

        # enrollment와 object간의 interaction
        one_enroll_interaction = one_log_all[['enroll_id', 'type', 'type_id']].groupby(['enroll_id', 'type', 'type_id']).nunique().reset_index()
        two_enroll_interaction = two_log_all[['enroll_id', 'type', 'type_id']].groupby(['enroll_id', 'type', 'type_id']).nunique().reset_index()
        third_enroll_interaction = third_log_all[['enroll_id', 'type', 'type_id']].groupby(['enroll_id', 'type', 'type_id']).nunique().reset_index()

        object_info = log_all[['course_id', 'type_id', 'type']].groupby(['course_id', 'type_id', 'type']).nunique().reset_index()

        object_feature = dict()
        object_event = log_all[['type', 'type_id']].groupby(['type', 'type_id']).nunique().reset_index()
        for index, datarow in tqdm(object_event.iterrows(), total=object_event.shape[0]):
            feature = np.zeros(4)
            feature[int(datarow['type'])] = 1
            object_feature[object_node_id[datarow['type_id']]] = feature
            
        e_in_cousre_dic = defaultdict(list)

        labels_df = pd.merge(labels_df, user_info[['id', 'course_id', 'sec_user_id']].rename(columns={'id':'enroll_id', 'sec_user_id':'user_id'}), how='left', on='enroll_id')
        for course in course_list:
            for enroll in labels_df[labels_df['course_id'] == course]['enroll_id']:
                e_in_cousre_dic[course_node_id[course]].append(enroll_node_id[enroll])
            course_feature[course] = np.append(course_feature[course], [len(e_in_cousre_dic[course_node_id[course]])])
            
        stan_scaler = StandardScaler()

        col_name = []
        for i in range(30):
            col_name.append(f'clicksum_day{i}')
            
        enroll_feature_df = pd.DataFrame(enroll_feature, index=col_name).T
        enroll_feature_df = enroll_feature_df.reset_index().rename(columns={'index': 'enroll_id'})
        enroll_feature_df = pd.merge(enroll_feature_df, log_all[['enroll_id', 'course_id']].groupby(['enroll_id', 'course_id']).first().reset_index(), how='left', on='enroll_id').set_index('enroll_id', drop=True)
        
        node_feature_df = pd.DataFrame()
        for course_id in course_node_id.keys():
            stan_df = enroll_feature_df[enroll_feature_df['course_id'] == course_id].drop('course_id', 1)
            index_name = stan_df.index.tolist()
            stan_df = pd.DataFrame(stan_scaler.fit_transform(stan_df), index = index_name, columns = col_name)
            stan_df['course_id'] = course_id
            node_feature_df = pd.concat([node_feature_df, stan_df])
            
        course_feature_df = pd.DataFrame(course_feature, index=course_cate_list+['total_enrollment']).T
        index_name = course_feature_df.index.tolist()
        course_feature_df = pd.DataFrame(stan_scaler.fit_transform(course_feature_df), index = index_name, columns = course_cate_list+['total_enrollment'])

        train_df = pd.merge(node_feature_df.reset_index().rename(columns={'index':'enroll_id'}), course_feature_df.reset_index().rename(columns={'index':'course_id'}), how='left', on='course_id').drop('course_id', 1)
        train_df = pd.merge(train_df, labels_df, how='left', on='enroll_id')

        X_train, X_test, y_train, y_test = train_test_split(train_df.set_index('enroll_id')[['clicksum_day0', 'clicksum_day1', 'clicksum_day2', 'clicksum_day3',
            'clicksum_day4', 'clicksum_day5', 'clicksum_day6', 'clicksum_day7',
            'clicksum_day8', 'clicksum_day9', 'clicksum_day10', 'clicksum_day11',
            'clicksum_day12', 'clicksum_day13', 'clicksum_day14', 'clicksum_day15',
            'clicksum_day16', 'clicksum_day17', 'clicksum_day18', 'clicksum_day19',
            'clicksum_day20', 'clicksum_day21', 'clicksum_day22', 'clicksum_day23',
            'clicksum_day24', 'clicksum_day25', 'clicksum_day26', 'clicksum_day27',
            'clicksum_day28', 'clicksum_day29', 'LECTURE', 'QUIZ', 'COMMENT',
            'FORUM', 'INFOPAGE', 'INTRODUCTION', 'total_enrollment']], train_df.set_index('enroll_id')[['labels3']], test_size=0.40, random_state=321)

        X_train.to_pickle('data/X_train_df_Naver.pkl')
        X_test.to_pickle('data/X_test_df_Naver.pkl')
        y_train.to_pickle('data/y_train_df_Naver.pkl')
        y_test.to_pickle('data/y_test_df_Naver.pkl')

        #-------------------------------make Graph------------------------------
        print('Graph building')
        
        src_node, dst_node = [], []
        one_etype, two_etype, third_etype = [], [], []
        one_node_feature, two_node_feature, third_node_feature = [], [], []
        non_feature = []
        o_in_cousre_dic = defaultdict(list) 
        one_e_in_object_dic, one_o_in_enroll_dic, one_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        two_e_in_object_dic, two_o_in_enroll_dic, two_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        third_e_in_object_dic, third_o_in_enroll_dic, third_student_eid_dic = defaultdict(list), defaultdict(list), defaultdict(list)
        e_with_student_dic = dict()

        # Make Object-Course Edge
        for index, data_row in tqdm(object_info.iterrows(), total=object_info.shape[0]):
            src_node.append(object_node_id[data_row['type_id']])
            dst_node.append(course_node_id[data_row['course_id']])
            
            if data_row['type'] == 0: # LECTURE
                one_etype.append(0)
                two_etype.append(0)
                third_etype.append(0)
            if data_row['type'] == 1: # QUIZ
                one_etype.append(1)
                two_etype.append(1)
                third_etype.append(1)
            if data_row['type'] == 2: # DISCUSSION
                one_etype.append(2)
                two_etype.append(2)
                third_etype.append(2)
            if data_row['type'] == 3: # ETC
                one_etype.append(3)
                two_etype.append(3)
                third_etype.append(3)
            
            o_in_cousre_dic[course_node_id[data_row['course_id']]].append(object_node_id[data_row['type_id']])
                
        # Make Enrollemt-Object Edge
        one_src_node, one_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(one_enroll_interaction.iterrows(), total = one_enroll_interaction.shape[0]):
            one_src_node.append(enroll_node_id[data_row['enroll_id']])
            one_dst_node.append(object_node_id[data_row['type_id']])
            
            if data_row['type'] == 0: # LECTURE
                one_etype.append(4)
            if data_row['type'] == 1: # QUIZ
                one_etype.append(5)
            if data_row['type'] == 2: # DISCUSSION
                one_etype.append(6)
            if data_row['type'] == 3: # ETC
                one_etype.append(7)
                
            one_e_in_object_dic[object_node_id[data_row['type_id']]].append(enroll_node_id[data_row['enroll_id']])
            one_o_in_enroll_dic[enroll_node_id[data_row['enroll_id']]].append(object_node_id[data_row['type_id']])
            
        two_src_node, two_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(two_enroll_interaction.iterrows(), total = two_enroll_interaction.shape[0]):
            two_src_node.append(enroll_node_id[data_row['enroll_id']])
            two_dst_node.append(object_node_id[data_row['type_id']])

            if data_row['type'] == 0: # LECTURE
                two_etype.append(4)
            if data_row['type'] == 1: # QUIZ
                two_etype.append(5)
            if data_row['type'] == 2: # DISCUSSION
                two_etype.append(6)
            if data_row['type'] == 3: # ETC
                two_etype.append(7)
                
            two_e_in_object_dic[object_node_id[data_row['type_id']]].append(enroll_node_id[data_row['enroll_id']])
            two_o_in_enroll_dic[enroll_node_id[data_row['enroll_id']]].append(object_node_id[data_row['type_id']])
            
        third_src_node, third_dst_node = src_node.copy(), dst_node.copy()
        for index, data_row in tqdm(third_enroll_interaction.iterrows(), total = third_enroll_interaction.shape[0]):
            third_src_node.append(enroll_node_id[data_row['enroll_id']])
            third_dst_node.append(object_node_id[data_row['type_id']])

            if data_row['type'] == 0: # LECTURE
                third_etype.append(1)
            if data_row['type'] == 1: # QUIZ
                third_etype.append(2)
            if data_row['type'] == 2: # DISCUSSION
                third_etype.append(3)
            if data_row['type'] == 3: # ETC
                third_etype.append(4)
                
            third_e_in_object_dic[object_node_id[data_row['type_id']]].append(enroll_node_id[data_row['enroll_id']])
            third_o_in_enroll_dic[enroll_node_id[data_row['enroll_id']]].append(object_node_id[data_row['type_id']])
            
        one_src_node, one_dst_node = one_src_node + one_dst_node, one_dst_node + one_src_node
        two_src_node, two_dst_node = two_src_node + two_dst_node, two_dst_node + two_src_node
        third_src_node, third_dst_node = third_src_node + third_dst_node, third_dst_node + third_src_node
        one_etype = one_etype + one_etype
        two_etype = two_etype + two_etype
        third_etype = third_etype + third_etype
        
        # Make Student Edge
        for student in tqdm(list(set(labels_df['user_id']))):
            student_list_df = labels_df[labels_df['user_id']==student]
            student_list = list(map((lambda x: enroll_node_id[x]), student_list_df['enroll_id'].tolist()))
            # student_truth = list(map((lambda x: enroll_truth[x]), student_list_df['enrollment_id'].tolist()))
            
            student_truth1 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels1'].values[0]), student_list_df['enroll_id'].tolist()))
            student_truth2 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels2'].values[0]), student_list_df['enroll_id'].tolist()))
            student_truth3 = list(map((lambda x: labels_df[labels_df['enroll_id'] == x]['labels3'].values[0]), student_list_df['enroll_id'].tolist()))
            
            e_with_student_dic[student] = student_list
            
            n = len(student_list)
            for src_student in range(n):
                for dst_student in range(n):
                    if src_student != dst_student:
                        if student_truth1[src_student] == 1:
                            one_etype.append(5)
                        else: # dropout
                            one_etype.append(6)
                        if student_truth2[src_student] == 1:
                            two_etype.append(5)
                        else: # dropout
                            two_etype.append(6)
                        if student_truth3[src_student] == 1:
                            third_etype.append(5)
                        else: # dropout
                            third_etype.append(6)
                        
                        one_student_eid_dic[student_list[src_student]].append(len(one_src_node))
                        two_student_eid_dic[student_list[src_student]].append(len(two_src_node))
                        third_student_eid_dic[student_list[src_student]].append(len(third_src_node))
                            
                        one_src_node.append(student_list[src_student])
                        one_dst_node.append(student_list[dst_student])
                        two_src_node.append(student_list[src_student])
                        two_dst_node.append(student_list[dst_student])
                        third_src_node.append(student_list[src_student])
                        third_dst_node.append(student_list[dst_student])
            
        one_graph = dgl.graph((one_src_node, one_dst_node), num_nodes = num_node)
        two_graph = dgl.graph((two_src_node, two_dst_node), num_nodes = num_node)
        third_graph = dgl.graph((third_src_node, third_dst_node), num_nodes = num_node)

        #-------------------------------ADD Node Feature------------------------------
        enroll_zero_list = np.zeros(10)
        cousre_zero_list = np.zeros(len(course_cate_list)+1)
        for e_node_id in tqdm(enroll_node_id.keys()):
            one_feature = np.array([1,0,0])
            two_feature = np.array([1,0,0])
            third_feature = np.array([1,0,0])
            
            one_feature = np.append(one_feature, node_feature_df.loc[e_node_id].tolist()[0:10])
            two_feature = np.append(two_feature, node_feature_df.loc[e_node_id].tolist()[10:20])
            third_feature = np.append(third_feature, node_feature_df.loc[e_node_id].tolist()[20:30])
            
            one_feature = np.append(one_feature, cousre_zero_list)
            two_feature = np.append(two_feature, cousre_zero_list)
            third_feature = np.append(third_feature, cousre_zero_list)
            
            one_node_feature.append(one_feature)
            two_node_feature.append(two_feature)
            third_node_feature.append(third_feature)
            
            non_feature.append([1,0,0])
            
        for o_node_id in tqdm(object_node_id.values()):
            feature = np.array([0,1,0])
            
            feature = np.append(feature, enroll_zero_list)
            feature = np.append(feature, cousre_zero_list)
            
            one_node_feature.append(feature)
            two_node_feature.append(feature)
            third_node_feature.append(feature)
            
            non_feature.append([0,1,0])
            
        for c_node_id in tqdm(course_node_id.keys()):
            feature = np.array([0,0,1])
            
            feature = np.append(feature, enroll_zero_list)
            feature = np.append(feature, course_feature_df.loc[c_node_id].tolist())
            
            one_node_feature.append(feature)
            two_node_feature.append(feature)
            third_node_feature.append(feature)
            
            non_feature.append([0,0,1])
            
        one_graph.ndata['feature'] = torch.FloatTensor(one_node_feature)
        two_graph.ndata['feature'] = torch.FloatTensor(two_node_feature)
        third_graph.ndata['feature'] = torch.FloatTensor(third_node_feature)

        one_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)
        two_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)
        third_graph.ndata['non_feature'] = torch.FloatTensor(non_feature)

        one_graph.edata['etype'] = torch.tensor(one_etype)
        two_graph.edata['etype'] = torch.tensor(two_etype)
        third_graph.edata['etype'] = torch.tensor(third_etype)

        self.one_graph = one_graph
        self.two_graph = two_graph
        self.third_graph = third_graph
        self.labels_df = labels_df
        self.enroll_node_id = enroll_node_id
        self.object_node_id = object_node_id
        self.course_node_id = course_node_id
        self.e_in_cousre_dic = e_in_cousre_dic
        self.o_in_cousre_dic = o_in_cousre_dic
        self.e_with_student_dic = e_with_student_dic
        
        self.one_student_eid_dic = one_student_eid_dic
        self.two_student_eid_dic = two_student_eid_dic
        self.third_student_eid_dic = third_student_eid_dic
        
        self.one_o_in_enroll_dic = one_o_in_enroll_dic
        self.two_o_in_enroll_dic = two_o_in_enroll_dic
        self.third_o_in_enroll_dic = third_o_in_enroll_dic
        self.one_e_in_object_dic = one_e_in_object_dic
        self.two_e_in_object_dic = two_e_in_object_dic
        self.third_e_in_object_dic = third_e_in_object_dic
        
    def __getitem__(self, i):
        if i == 0:
            return self.one_graph
            
        target_enroll = self.enroll_node_id[i]
        enroll_info = self.labels_df[self.labels_df['enroll_id'] == i]
        target_course = self.course_node_id[enroll_info['course_id'].values[0]]
        student_name = enroll_info['user_id'].values[0]
        
        o_with_target = self.o_in_cousre_dic[target_course]
        enroll_list = self.e_with_student_dic[student_name]
                
        one_subgraph = get_subgraph(graph = self.one_graph,
                                    o_with_target=torch.tensor(o_with_target),  
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        two_subgraph = get_subgraph(graph = self.two_graph,
                                    o_with_target=torch.tensor(o_with_target),  
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        third_subgraph = get_subgraph(graph = self.third_graph,
                                    o_with_target=torch.tensor(o_with_target), 
                                    e_with_student=torch.tensor(enroll_list, dtype=torch.int64), 
                                    target_course=torch.tensor([target_course], dtype=torch.int64),
                                    target_enroll=torch.tensor([target_enroll], dtype=torch.int64),
                                    )
        
        target_mask = np.zeros(one_subgraph.num_nodes())
        target_enroll_id = (one_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (one_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        one_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        target_mask = np.zeros(two_subgraph.num_nodes())
        target_enroll_id = (two_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (two_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        two_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        target_mask = np.zeros(third_subgraph.num_nodes())
        target_enroll_id = (third_subgraph.ndata[dgl.NID] == target_enroll).nonzero(as_tuple=True)[0]
        target_course_id = (third_subgraph.ndata[dgl.NID] == target_course).nonzero(as_tuple=True)[0]
        target_mask[target_enroll_id] = 1
        target_mask[target_course_id] = 2
        third_subgraph.ndata['target'] = torch.tensor(target_mask)
        
        # Masking student edge
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.one_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (one_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: one_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((one_subgraph.edges()[0] == e_id) & (one_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        one_subgraph.edata['etype'][drop_etype_tensor.tolist()] = one_subgraph.edata['etype'][new_etype]
        
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.two_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (two_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: two_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((two_subgraph.edges()[0] == e_id) & (two_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        two_subgraph.edata['etype'][drop_etype_tensor.tolist()] = two_subgraph.edata['etype'][new_etype]
        
        drop_etype_tensor =torch.tensor([])
        for edge_id in self.third_student_eid_dic[target_enroll]:
            drop_etype_tensor = torch.cat([drop_etype_tensor, (third_subgraph.edata[dgl.EID] == edge_id).nonzero(as_tuple=True)[0]])
        e_list = list(map(lambda x: third_subgraph.edges()[1][int(x)].item(), drop_etype_tensor.tolist()))  
        new_etype = []
        for e_id in e_list:
            new_etype.append(((third_subgraph.edges()[0] == e_id) & (third_subgraph.edges()[1] == target_enroll_id)).nonzero(as_tuple=True)[0].item())
        third_subgraph.edata['etype'][drop_etype_tensor.tolist()] = third_subgraph.edata['etype'][new_etype]
        
        return one_subgraph, two_subgraph, third_subgraph, torch.Tensor(self.labels_df[self.labels_df['enroll_id']==i].values[-1][1:5])
        