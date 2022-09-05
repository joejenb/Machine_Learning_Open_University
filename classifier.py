import os
import zipfile
import pandas as pd
import numpy as np
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm


'''Before running any code all of the above libraries must be installed.
To load, prepare and predict on the training and test set simply run the code. Please bear in mind the initial data processing is very lengthy.'''

OPEN_LEARNING_PATH = os.path.join("datasets", "open_learning")
OPEN_LEARNING_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"

def fetch_courses_data(open_learning_url=OPEN_LEARNING_URL, open_learning_path=OPEN_LEARNING_PATH):
    ''' Downloads, unzips and saves all the tables/csv files provided by the open learning data set, with corresponding names'''
    if not os.path.isdir(open_learning_path):
        os.makedirs(open_learning_path)
    zip_path = os.path.join(open_learning_path, "anonymisedData.zip")
    urllib.request.urlretrieve(open_learning_url, zip_path)
    open_learning_zip = zipfile.ZipFile(zip_path)
    open_learning_zip.extractall(path=open_learning_path)
    open_learning_zip.close()


def load_courses_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the courses table, stored in the file courses.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "courses.csv")
    return pd.read_csv(csv_path)


def load_studentInfo_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the studentInfo table, stored in the file studentInfo.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "studentInfo.csv")
    return pd.read_csv(csv_path)

def load_studentVle_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the studentVle table, stored in the file studentVle.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "studentVle.csv")
    return pd.read_csv(csv_path)


def load_assessments_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the assessments table, stored in the file assessments.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "assessments.csv")
    return pd.read_csv(csv_path)


def load_studentAssessment_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the studentAssessments table, stored in the file studentAssessments.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "studentAssessment.csv")
    return pd.read_csv(csv_path)


def load_studentRegistration_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the studentRegistration table, stored in the file studentRegistration.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "studentRegistration.csv")
    return pd.read_csv(csv_path)


def load_vle_data(data_path=OPEN_LEARNING_PATH):
    '''Loads the data for the vle table, stored in the file vle.csv, and stores it in a pandas dataframe'''
    csv_path = os.path.join(data_path, "vle.csv")
    return pd.read_csv(csv_path)

def split_data(studentInfo_data):
    '''Uses stratified sampling on the previously highest education level of a student to split the studentInfo table into a
    train(80%) and test(20%) set. Sets 'random_state' to be 42, ensuring that the split is exactly the same each time the dataset is loaded.'''
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(studentInfo_data, studentInfo_data["highest_education"]):
        strat_train_set = studentInfo_data.loc[train_index]
        strat_test_set = studentInfo_data.loc[test_index]
    return strat_train_set, strat_test_set

def calc_tma_per():
    '''Although not used as one of the final features, calculates the percentage of each module presentation that is assesed with teacher
    marked assessments'''
    module_data = pd.DataFrame(columns=["code_module", "code_presentation", "per_TMA"])
    for index, mod in courses_data.iterrows():
        mod_ass = assessment_data.loc[mod["code_module"] == assessment_data["code_module"]]
        mod_ass = mod_ass.loc[mod["code_presentation"] == mod_ass["code_presentation"]]
        tma = mod_ass.loc["TMA" == mod_ass["assessment_type"]]["weight"].sum()
        all_der = pd.DataFrame(data={"code_module": [mod["code_module"]], "code_presentation": [mod["code_presentation"]], "per_TMA": [tma]})
        module_data = pd.concat([module_data, all_der], ignore_index=True)
    return module_data

def calc_relevant_sites(num_sites):
    '''Identifies a certain number of the resources (sepcified by num_sites) that have the most total clicks, given that they were only visited 
    by less than the average number of students per presentation. Creates a new column for each of these most popular resources in the studentInfo 
    dataframe, the values of which are the number of times a student has clicked on that resource.'''
    
    relevant_sites = pd.DataFrame(columns=["id_site", "num_students", "clicks_per_site"])
    student_per_pres = pd.DataFrame(columns = ["code_module", "code_presentation", "no_students"])
    
    # Reduce to table that only contains sites not visited by everyone in a presentation but visited frequently by those that do visit them
    # Frequently being more than 6 times
    # First find number of students per presentation
    
    for index, mod in courses_data.iterrows():
        mod_ass = studentInfo_data.loc[mod["code_module"] == studentInfo_data["code_module"]]
        mod_ass = mod_ass.loc[mod["code_presentation"] == mod_ass["code_presentation"]]
        all_der = pd.DataFrame(data={"code_module": [mod["code_module"]], "code_presentation": [mod["code_presentation"]], "no_students": [len(mod_ass)]})
        student_per_pres = pd.concat([student_per_pres, all_der], ignore_index=True)
        
    avg_student_per_pres = student_per_pres["no_students"].mean()
    
    for index, mod in vle_data.iterrows():
        mod_vle = student_vle_data.loc[mod["id_site"] == student_vle_data["id_site"]]
        num_clicks = mod_vle["sum_click"].sum()
        mod_vle = mod_vle.drop_duplicates(["id_student", "code_module", "code_presentation"], keep="first")
        if len(mod_vle) < avg_student_per_pres and num_clicks>1000:
            all_der = pd.DataFrame(data={"id_site": [mod["id_site"]], "num_students": [len(mod_vle)], "clicks_per_site": num_clicks})
            relevant_sites = pd.concat([relevant_sites, all_der], ignore_index = True)

    relevant_sites = relevant_sites.sort_values(by="clicks_per_site", ascending=False)
    site_names = [site for site in relevant_sites.head(num_sites)["id_site"]]
    return pd.DataFrame(columns=["id_student", "code_module", "code_presentation"]+site_names)
        
class LabelBinarizer(TransformerMixin):
    '''Utilises scikit learns MultiLabelBianrizer such that categorical attributes that are fitted are first converted into integer representations,
    these are then one hot encoded to prevent any inference of association between similar values'''
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)
        
    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self
    
    def transform(self, x, y=None):
        return self.encoder.transform(x)
    
class final_selector(BaseEstimator, TransformerMixin):
    '''Used to select the final numerical and categorical attributes that will be used for input by dropping those not needed'''
    def __init__(self, to_drop):
        self.to_drop = to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        final_attribs = list(set(X[1]) - set(self.to_drop))
        X = pd.DataFrame(X[0], columns = X[1])
        return X[final_attribs].values
    
class custom_imputer(BaseEstimator, TransformerMixin):
    '''Used to fill in missing imd band values in the data set. Does this using the median imd band for the instances corresponing 
    region'''
    def __init__(self, attribs, method = "median"):
        self.method = method
        self.attribs = attribs
        self.imd_bands = ["{}-{}%".format(x, x+10) for x in range(0, 100, 10)]
        self.imd_ranks = dict(zip(self.imd_bands, range(len(self.imd_bands))))
        self.imd_avgs = {}
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns = self.attribs)
        missing_imd = X.isnull().any(axis=1)
        X["imd_band_rank"] = X["imd_band"].map(self.imd_ranks)
        for index, stu in X[missing_imd].iterrows():
            if stu["imd_band"] not in self.imd_bands:
                    if stu["region"] not in self.imd_avgs.keys():
                        region = X.loc[X["region"] == stu["region"]].dropna()
                        region_ranked = region.sort_values(['imd_band_rank'])
                        X.loc[index, "imd_band"] = region_ranked.reset_index()["imd_band"][len(region_ranked)//2]
                        self.imd_avgs[stu["region"]] = X["imd_band"][index]
                    else:
                        X.loc[index, "imd_band"] = self.imd_avgs[stu["region"]]
        X = X.drop("imd_band_rank", axis=1)
        X["imd_band"] = X["imd_band"].astype(str)
        return X.values, X.columns
    
class unique_attributes_adder(BaseEstimator, TransformerMixin):
    '''Used to add atributes found in other tables, those being the percentage of assessments that are
    teacher marked for a module presentation and duration of each module presentation'''
    def __init__(self, attribs, next_attribs):
        self.attribs = attribs
        self.next_attribs = next_attribs
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns = self.attribs)
        if 'per_TMA' in self.next_attribs:
            per_tma = calc_tma_per()
            X = pd.merge(X, per_tma, on=['code_module', 'code_presentation'], how='outer')
        if 'module_presentation_length' in self.next_attribs:
            X = pd.merge(X, courses_data, on=['code_module', 'code_presentation'], how='outer')
        return X.values
    
class derived_attributes_adder(BaseEstimator, TransformerMixin):
    '''Adds numerical attributes, that are derived from others, to the table. These include the average time between which a student makes 
    a click, the total number of clicks made by a student, the number of different resources that a student visits and the number of clicks
    a student makes on each of the top sites(identified by 'calc_relevant_sites')'''
    def __init__(self, attribs, next_attribs, top_sites=20):
        self.attribs = attribs
        self.new_attribs = list(set(next_attribs) - set(attribs))
        self.top_sites = top_sites
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns = self.attribs)
        der_data = pd.DataFrame()
        
        if 'clicks_per_site' in self.new_attribs:
            site_names = calc_relevant_sites(self.top_sites)
            
        for index, mod in X.iterrows():
            new_values = {"id_student": mod["id_student"], "code_module": mod["code_module"], "code_presentation": mod["code_presentation"]}
            stu_tbl = student_vle_data.loc[student_vle_data["id_student"] == mod["id_student"]]
            stu_tbl = stu_tbl.loc[stu_tbl["code_module"] == mod["code_module"]]
            stu_tbl = stu_tbl.loc[stu_tbl["code_presentation"] == mod["code_presentation"]]
            ord_stu = stu_tbl.sort_values(by="date")
            
            if 'avg_time' in self.new_attribs:
                new_values['avg_time'] = [ord_stu["date"].diff().mean()]
                if np.isnan(new_values['avg_time']):
                    avg_dif = courses_data.loc[(courses_data['code_module'] == mod["code_module"]) & (courses_data['code_presentation'] == mod["code_presentation"])]
                    new_values['avg_time'] = [avg_dif.reset_index().at[0,"module_presentation_length"]]
                    
            if 'all_clicks' in self.new_attribs:
                new_values['all_clicks'] = [stu_tbl["sum_click"].sum()]
                
            if 'all_sites' in self.new_attribs:
                new_values['all_sites'] = [len(stu_tbl.drop_duplicates(subset="id_site", keep="first"))]
                
            if 'clicks_per_site' in self.new_attribs:
                for site in site_names.columns:
                    if type(site) == int:
                        new_values[str(site)] = [stu_tbl.loc[stu_tbl["id_site"] == site]["sum_click"].sum()]
                    
            all_der = pd.DataFrame.from_dict(new_values)
            if der_data.empty:
                der_data = pd.DataFrame.from_dict(new_values)
            else:
                der_data = pd.concat([der_data,  all_der], ignore_index=True)
        X = pd.merge(X, der_data, on=["id_student", "code_module", "code_presentation"], how='outer')
        return X.values, X.columns
            
'''Load in all the needed data provided by open learning'''
fetch_courses_data()
courses_data = load_courses_data()
assessment_data = load_assessments_data()
studentInfo_data = load_studentInfo_data()
student_vle_data = load_studentVle_data()
vle_data = load_vle_data()

'''Split the data into a test and train set'''
studentInfo_train, studentInfo_test = split_data(studentInfo_data)

'''Split the training and test data into respective input and output features'''
full_data = studentInfo_train.drop("final_result", axis=1)
full_results = pd.DataFrame(studentInfo_train["final_result"].values, columns=["final_result"])
test_data = studentInfo_test.drop("final_result", axis=1)
test_results = pd.DataFrame(studentInfo_test["final_result"].values, columns=["final_result"])

'''Specify features that will originally be included, allowing for others to be derived'''
init_attribs = list(full_data.columns)
cat_attribs = ["code_module", "code_presentation", "highest_education", "gender", "region", "imd_band", "age_band", "disability"]
num_attribs = list(set(list(init_attribs))-set(cat_attribs))

'''Specify what derived features should actually be added'''
uniq_attribs = init_attribs + []#'per_TMA', 'module_presentation_length'
der_attribs = uniq_attribs + ['all_clicks', 'clicks_per_site', 'avg_time', 'all_sites']

'''Specify what features must be removed, and in those that will be used, once the data has been fully processed'''
to_drop = ['gender', 'region']
num_to_drop = to_drop + cat_attribs
cat_to_drop = to_drop + num_attribs
#final_attribs = list(set(der_attribs) - set(to_drop))
#final_cat_attribs = list(set(final_attribs) & set(init_cat_attribs))
#final_num_attribs = list(set(final_attribs) & set(init_num_attribs))

'''Pipeline for the processing of numerical data derives and selects attributes that  will be used, then standardises these using
sklearns standScaler'''
num_pipeline = Pipeline([('uniq_attribs_adder', unique_attributes_adder(init_attribs, uniq_attribs)),
                         ('der_attribs_adder', derived_attributes_adder(uniq_attribs, der_attribs)),
                         ('final_selector', final_selector(num_to_drop)),
                         ('scaler', StandardScaler()),
                        ])

'''Pipeline for the processing of categorical attributes, imputes missing values in the data set, selects the attributes to be used 
and then one hot encodes the categories using sklearns LabelBinarizer'''
cat_pipeline = Pipeline([('imputer', custom_imputer(init_attribs)),
                         ('final_selector', final_selector(cat_to_drop)),
                         ('cat_binarizer', LabelBinarizer()),
                        ])

'''Pipeline that is responsible for joining the outputs of both the number and category pipelines, giving a table of the fully prepared input data'''
full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                               ("cat_pipeline", cat_pipeline),
                                              ])

prepared_data = full_pipeline.fit_transform(full_data.values)
prepared_test_data = full_pipeline.fit_transform(test_data.values)

def validate_and_scores(classifier, in_data, res_data, fit_data = False):
    '''Carries out 10 fold cross validation on input classifier, handles cases where the data used to fit 
    the first classifier is the same as that used to fit the second and where it is not'''
    if type(fit_data) != bool:
        skfolds = StratifiedKFold(n_splits=10, random_state=42)
        total_score = 0
        for train_index, test_index in skfolds.split(in_data, res_data):
            clone_clf = clone(classifier)
            in_train_folds = in_data[train_index]
            fit_train_folds = fit_data[train_index]
            out_train_folds = (res_data[train_index])
            in_test_fold = in_data[test_index]
            out_test_fold = (res_data[test_index])
            clone_clf.fit(in_train_folds, fit_train_folds)
            clf_predict = clone_clf.predict(in_test_fold)
            n_correct = sum(clf_predict == out_test_fold)
            total_score += n_correct / len(clf_predict)
        average = total_score/10
        
    else:
        scores = cross_validate(classifier, in_data, res_data, scoring=("accuracy"), cv=10)#, "average_precision", "recall"
        average = scores["test_score"].mean()
    print("Means:", "Accuracy: ", average)#, '\nPrescision: ', scores["test_average_precision"].mean())
    return average
    

def parameter_search(classifier, in_data, res_data, params_1, params_2, fit_data = False):
    '''Given a set of possible parameters for each classifier, performs grid search on both as a whole. Uses validate_and_scores
    function to assess the accuracy of each combination of parameters'''
    best_score = 0
    all_comb = {"grid_1": list(ParameterGrid(params_1)), "grid_2": list(ParameterGrid(params_2))}
    for params in ParameterGrid(all_comb):
        clone_clf = clone(classifier)
        clone_clf.set_params(params)
        score = validate_and_scores(clone_clf, in_data, res_data, fit_data)
        if score > best_score:
            best_score = score.copy()
            best_grid = params
            
    print("Best score and grid: " + str(best_score) + "  " + str(best_grid))

class OVO_SVM_and_SGD(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ovo_svm = svm.SVC()
        self.sgd_cls = SGDClassifier(random_state=42)

    def fit(self, in_data, fit_data):
        '''Fits the one vs one SVM with the data for all four outcome types and then uses it to predict the input. Output
        of this is used along with the original input data to fit the SGD binary classifier'''
        self.set_results(fit_data)
        self.ovo_svm.fit(in_data, self.four_res)
        ovo_pred = self.ovo_svm.predict(in_data)
        imp_in_data = np.concatenate((in_data, np.atleast_2d(ovo_pred).T), axis=1)
        self.sgd_cls.fit(imp_in_data, self.bi_res)

    def predict(self, in_data):
        '''SVM makes a prediciton as to which of the four outcome types the input is. SGD then uses this along with the orginal
        input to give a binary classification'''
        ovo_pred = self.ovo_svm.predict(in_data)
        imp_in_data = np.concatenate((in_data, np.atleast_2d(ovo_pred).T), axis=1)
        return self.sgd_cls.predict(imp_in_data)

    def set_results(self, res_data):
        '''Takes input, four category data and converts it into only two possible outputs(Pass or Fail)'''
        bi_res = []
        for i in range(len(res_data)):
            if res_data[i] == 0 or res_data[i] == 1:
                bi_res.append(0)
            elif res_data[i] == 2 or res_data[i] == 3:
                bi_res.append(1)
        self.bi_res = np.asarray(bi_res)
        self.four_res = res_data.copy()
        print(len(self.bi_res), len(self.four_res))

    def set_params(self, params):
        '''Facilitates the setting of parameters for each classifier. Predominantely used during the parameter search
        process'''
        self.ovo_svm.set_params(**params["grid_1"])
        self.sgd_cls.set_params(**params["grid_2"])


class logistic_reg_and_rand_forest_bi(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_reg = LogisticRegression()
        self.ran_for_reg = RandomForestClassifier(n_estimators = 500)

    def fit(self, in_data, res_data):
        '''Fits the logistic regressor with output of only two types(Pass or Fail), carries out prediciton on input data and then
        uses this along with the original input to train the random forest'''
        self.set_results(res_data)
        self.log_reg.fit(in_data, self.bi_res)
        log_pred = self.log_reg.predict(in_data)
        imp_in_data = np.concatenate((in_data, np.atleast_2d(log_pred).T), axis=1)
        self.ran_for_reg.fit(imp_in_data, self.bi_res)

    def predict(self, in_data):
        '''Uses logistic regression to make initial prediction, random forest then uses this to inform its decision'''
        log_pred = self.log_reg.predict(in_data)
        imp_in_data = np.concatenate((in_data, np.atleast_2d(log_pred).T), axis=1)
        prediction = self.ran_for_reg.predict(imp_in_data)
        return prediction

    def predict_proba(self, in_data):
        log_pred = self.log_reg.predict(in_data)
        imp_in_data = np.concatenate((in_data, np.atleast_2d(log_pred).T), axis=1)
        prediction_prob = self.ran_for_reg.predict_proba(imp_in_data)
        print(prediction_prob)
        return prediction_prob

    def set_results(self, res_data):
        '''Takes binary output categories passed in and stores them as an attribute'''
        self.bi_res = np.copy(res_data)

    def set_params(self, params):
        '''Facilitates the setting of parameters for each classifier. Predominantely used during the parameter search
        process'''
        self.log_reg.set_params(**params["grid_1"])
        self.ran_for_reg.set_params(**params["grid_2"])

res_data = full_results.values
final_res = []
        
'''Converts text output categories to binary integer values that can then be used for training and testing'''
for i in range(len(res_data)):
    if res_data[i] == "Pass":
        final_res.append(0)
    elif res_data[i] == "Distinction":
        final_res.append(0)
    elif res_data[i] == "Fail":
        final_res.append(1)
    elif res_data[i] == "Withdrawn":
        final_res.append(1)
                
bi_class_res = np.asarray(final_res)

res_data = full_results.values
final_res = []

'''Converts text output categories to integer values that can then be used for training and testing'''
for i in range(len(res_data)):
    if res_data[i] == "Pass":
        final_res.append(0)
    elif res_data[i] == "Distinction":
        final_res.append(1)
    elif res_data[i] == "Fail":
        final_res.append(2)
    elif res_data[i] == "Withdrawn":
        final_res.append(3)

log_params = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-4, 4, 1), 'solver' : ['liblinear']}
rand_params = {'n_estimators' : list(range(10,101,20)), 'max_features' : list(range(6,32,10))}
svm_params = {'C': [50], 'gamma': [1,0.01],'kernel': ['rbf']}#0.1, , 'poly', 'sigmoid'
sgd_params = {'alpha': [1e-2], 'loss': ['log'], 'penalty': ['l2'], 'n_jobs': [-1]}#1e-4, 1e0, 1e2

four_class_res = np.asarray(final_res)
ovo_svm_and_sgd = OVO_SVM_and_SGD()
log_reg_and_rand_bi = logistic_reg_and_rand_forest_bi()

#parameter_search(ovo_svm_and_sgd, prepared_data, bi_class_res, svm_params, sgd_params, four_class_res)
#log_reg_and_rand_bi.fit(prepared_data, bi_class_res)

ovo_svm_and_sgd.fit(prepared_data, four_class_res)
ovo_pred = ovo_svm_and_sgd.predict(prepared_data)

log_reg_and_rand_bi.fit(prepared_data, bi_class_res)
log_pred = log_reg_and_rand_bi.predict(prepared_data)

ovo_pred = validate_and_scores(ovo_svm_and_sgd, prepared_data, bi_class_res, four_class_res)
log_pred = validate_and_scores(log_reg_and_rand_bi, prepared_data, bi_class_res)

'''Reduces test output categories to just two - Pass or Fail'''
def classes_to_bin(res_data):
    final_res = []
    for i in range(len(res_data)):
        if res_data[i] == "Pass":
            final_res.append(0)
        elif res_data[i] == "Distinction":
            final_res.append(0)
        elif res_data[i] == "Fail":
            final_res.append(1)
        elif res_data[i] == "Withdrawn":
            final_res.append(1)
    return final_res

bi_class_test_res = np.asarray(classes_to_bin(test_results.values.ravel()))
prediction_test_log = log_reg_and_rand_bi.predict(prepared_test_data)
prediction_test_ovo = ovo_svm_and_sgd.predict(prepared_test_data)
print(accuracy_score(bi_class_test_res, prediction_test_log))
print(accuracy_score(bi_class_test_res, prediction_test_ovo))
