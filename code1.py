file=open('data','r')
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import pandas as pd
text=pd.read_csv(file,sep="\t")
## ------------------------------------------------------------------------------------------------
## Data Cleaning

del text['start_time_DI']
del text['last_event_DI']
del text['userid_DI']
del text['roles']

text=text[text['incomplete_flag']!=1]
del text['incomplete_flag']

for i,row in text.iterrows():
    ## 1
    if(row[0]=="HarvardX/PH207x/2012_Fall"):
        text.set_value(i,0,'Health Statistics')
    ## 2
    if(row[0]=="HarvardX/CS50x/2012"):
        text.set_value(i,0,'Intro to CS')
    ## 3
    if(row[0]=="HarvardX/ER22x/2013_Spring"):
        text.set_value(i,0,'Justice')
    ## 4
    if(row[0]=="HarvardX/CB22x/2013_Spring"):
        text.set_value(i,0,'Ancient Greek Hero')
    ## 5
    if(row[0]=="HarvardX/PH278x/2013_Spring"):
        text.set_value(i,0,'Health Environment')

    ## 6
    if(row[0]=="MITx/3.091x/2012_Fall"):
        text.set_value(i,0,'Solid State Chemistry')
    ## 7
    if(row[0]=="MITx/14.73x/2013_Spring"):
        text.set_value(i,0,'Poverty')
    ## 8
    if(row[0]=="MITx/2.01x/2013_Spring"):
        text.set_value(i,0,'Structures-Civil')
    ## 9
    if(row[0]=="MITx/3.091x/2013_Spring"):
        text.set_value(i,0,'Solid State Chemistry')
    ## 10
    if(row[0]=="MITx/6.002x/2012_Fall"):
        text.set_value(i,0,'Circuits and Electronics')
    ## 11
    if(row[0]=="MITx/6.002x/2013_Spring"):
        text.set_value(i,0,'Circuits and Electronics')

    ## 12
    if(row[0]=="MITx/6.00x/2012_Fall"):
        text.set_value(i,0,'Computer Science')
    ## 13
    if(row[0]=="MITx/6.00x/2013_Spring"):
        text.set_value(i,0,'Computer Science')
    ## 14
    if(row[0]=="MITx/7.00x/2013_Spring"):
        text.set_value(i,0,'Biology')

    ## 15
    if(row[0]=="MITx/8.02x/2013_Spring"):
        text.set_value(i,0,'Electricity and Magnetism')
    ## 16
    if(row[0]=="MITx/8.MReV/2013_Summer"):
        text.set_value(i,0,'Mechanics Review')

del text['course_id']
text = text.rename(columns={0: 'Course_Name'})

##----------------------------------------------------------------------------------------------------------

## Predict best course based on ML
def predict_course(a,b,c,d):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    X = pd.DataFrame(text[['gender', 'YoB', 'LoE_DI', 'final_cc_cname_DI', 'Course_Name']])
    X = X.dropna(axis=0, how='any')

    data = pd.DataFrame(X[['gender', 'YoB', 'LoE_DI', 'final_cc_cname_DI']])
    labels = pd.DataFrame(X['Course_Name'])

    data["gender"] = data["gender"].astype('category')
    data["body_style_cat"] = data["gender"].cat.codes

    lb_make = LabelEncoder()
    data["make_code_1"] = lb_make.fit_transform(data["gender"])
    data["make_code_2"] = lb_make.fit_transform(data["YoB"])
    data["make_code_3"] = lb_make.fit_transform(data["LoE_DI"])
    data["make_code_4"] = lb_make.fit_transform(data["final_cc_cname_DI"])

    training = pd.DataFrame(data[['make_code_1', 'make_code_2', 'make_code_3', 'make_code_4']])

    X_train, X_test, y_train, y_test = train_test_split(training.values, labels.values, test_size=0.3)

    dt = DecisionTreeClassifier(max_leaf_nodes=5,max_depth=6)
    dt.fit(X_train, y_train)
    print("The accuracy of DT for predicting best course is : \n ")
    print(accuracy_score(y_test,dt.predict(X_test)))
    a=precision_recall_fscore_support(y_test, dt.predict(X_test), average='weighted')
    print("the fscore anf support ", a)
    # b=confusion_matrix(y_test,dt.predict((X_test)))
    # print("the confusion matrix is ",b)

    arr = [a,b,c,d]
    dynamic = pd.DataFrame(arr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dynamic_new = lb_make.fit_transform(dynamic)
        answer = dt.predict(dynamic_new)
        print("The course which many of your peers selected is ",answer)


## predict course based on ML done

## -----------------------------------------------------------------------------------------------------------

##  Certified
def certified(course):
    text_certified=text[text['certified']==1]
    text_registered=text[text['registered']==1]
    result_certified=text_certified.groupby('Course_Name')['certified']
    result_registered=text_registered.groupby('Course_Name')['registered']

    certified=result_certified.count()
    registered=result_registered.count()

    ratio=list(zip(certified,registered))
    percent_of_certified=[]
    for i in ratio:
        a=int(i[1])
        b=int(i[0])
        c=(b/a)*100
        # print(c)
        percent_of_certified.append(c)

    data_cert=list(zip(result_certified,percent_of_certified))

    cert_1=[]
    cert_2=[]
    for i in data_cert:
        # print(i[0][0],i[1])
        cert_1.append(i[0][0])
        cert_2.append(i[1])

    data_of_certified=list(zip(cert_1,cert_2))
    df=pd.DataFrame(data_of_certified)

     ## function parameter
    for i in df.values:
        if(i[0]==course):
            print(" *** The percentage of people who certified in " +course+ " course is ",i[1],"****")
## Certified function completed

##------------------------------------------------------------------------------------------------------------

## Drop outs function start
def dropouts(course):
    arr_course=[]
    arr_drops=[]

    for j in text.values:
        if(j[1]==0 and j[2]==0 and j[3]==0 and j[8]=="0" and  j[13]==0):
            arr_course.append(j[14])
            arr_drops.append(1)
        else:
            arr_course.append(j[14])
            arr_drops.append(0)
    drops=list(zip(arr_course,arr_drops,arr_drops))
    df_drop_outs=pd.DataFrame(drops)
    df_drop_outs.columns=['course','drops','counting']
    result=pd.DataFrame(df_drop_outs.groupby('course')['counting'].sum().reset_index())
    for i in result.values:
        if(i[0]==course):
            print("*** The drop outs of the course " + course + " are : ",i[1],"***")
## Drop out function ends

## -------------------------------------------------------------------------------------------------------------

## Grade Distribution function start
def grade_dist(course):
    X = pd.DataFrame(text[['Course_Name', 'grade', 'nevents', 'ndays_act', 'nplay_video', 'nchapters', 'nforum_posts']])

    X = X.dropna(axis=0, how='any')
    X_training = X[X['grade'] != "0"]
    X_testing = X[X['grade'] == "0"]
    train_data = X_training.values[:, 2:6]
    train_labels = X_training.values[:, 1]

    test_data = X_testing.values[:, 2:6]
    test_labels = X_testing.values[:, 1]

    dt = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=4)
    dt.fit(train_data, train_labels)
    y_pred_dt = dt.predict(test_data)
    print("Accuarcy of DT is : \n")
    print(accuracy_score(test_labels, y_pred_dt))

    array_grades = []
    array_courses = []

    for i in X.values:
        array_courses.append(i[0])

    j = 0
    for i in X.values:
        if (i[1] == '0'):
            array_grades.append(y_pred_dt[j])
            j = j + 1
        else:
            array_grades.append(i[1])

    grades_array = []
    try:
        for a in array_grades:
            i = float(a)
            if (i > 0 and i < 0.2):
                grades_array.append("D")
            if (i >= 0.2 and i < 0.5):
                grades_array.append("C")
            if (i >= 0.5 and i < 0.8):
                grades_array.append("B")
            if (i >= 0.8 and i <= 1):
                grades_array.append("A")
    except ValueError:
        pass

    grad_dist = list(zip(array_courses, grades_array,grades_array))

    df = pd.DataFrame(grad_dist)

    df.columns = ['course', 'grade','counting']
    res = pd.DataFrame(df.groupby(['course', 'grade'])['counting'].count().reset_index())
    print(" Grade distribution for " + course + " is : \n")
    for i in res.values:
        course=course
        if (i[0] == course):
            print(i[1], i[2])


## Grade distribution functin end
##-----------------------------------------------------------------------------------------------------------------

## Best Course
course1_KC=['classical greek civilization','model of humanity, the hero','socratic dialogue','plato memories','greek literature','2 forms of literature : epic and tragedy','7 tradegies','Homeric Iliad and Odyssey','The bacchic women','On heroes']
course2_KC=['computer programming','fundamentals of Computer science','solving programming problems','thinking algorithmically','abstraction','algorithms','data structures','encapsulation','resource management','resource management','security','software engineering','web development','familiarity to programming language','C','C++','Java','Python','SQL','JavaScript','HTML'] #Intro to comp sci CSC50X
course3_KC=['affirmative action', 'income distribution', 'same-sex marriage', 'the role of markets', 'debates about rights (human rights and property rights)', 'arguments for and against equality','dilemmas of loyalty in public and private life']
course4_KC=['How to analyze data sets using modern quantitative methods','How to discover patterns and extract knowledge from health data','The principles of biostatistics and epidemiology','public health and clinical research','outcomes measurement','study design options -bias and confounding','probability and diagnostic tests','confidence intervals','hypothesis testing','power and sample size determinations','life tables and survival methods','regression methods (both, linear and logistic)','sample survey techniques']
course5_KC=['Evidence of a changing climate','climate change effects on health',' Biodiversity and health',' Sustaining Life: How Human Health Depends on Biodiversity','The importance of biodiversity to health','how is biodiversity threatened','Biodiversity and food production',' Earth-Solar Energy Balance: a return to climate science','NOAA Annual Greenhouse gas Index','Changes in Atmospheric Constituents and in Radiative Forcing','Climate stabilization wedges ',' Technologies for Curbing CO2 emissions','Stabilization Wedges','Solving the Climate Problem','Hopeful examples of people, governments and corporations offering solutions']
course6_KC=['What is a poverty trap?','Food I: Is there a nutrition-based poverty trap?','Food II: The hidden traps','Delivering healthcare','Low-hanging fruit','How to make schools work for the poor','Beyond supply and demand wars','Can evidence play a role in the fight against poverty?']
course7_KC=['free body diagrams to formulate equilibrium equations','geometric constraints to formulate compatibility equations','Understand the concepts of stress and strain at a material point','internal stress and strain fields in the loaded elements','deformation in the loaded elements']
course8_KC=['the basic principles of the chemical bond','properties of solids','stiffness','electrical conductivity','thermal expansion','strength','chemical intuition','chemical principles','crystal structure and its relationship to properties','conductivity','optical transmission','stiffness','thermal expansion','strength','electronic structure','chemical bonding','atomic order and arrangements']
course9_KC=['design and analyze circuits','superposition','Thevenin method','lumped circuit models','abstraction to simplify circuit analysis','intuition to solve circuits','Construction of simple digital gates using MOSFET transistors','Measurement of circuit variables','virtual oscilloscopes','virtual multimeters','virtual signal generators']
course10_KC=['Computational Thinking and Data Science','A Notion of computationThe Python programming language','Some simple algorithmsTesting','debugging','An informal introduction to algorithmic complexity','Data structures']
course11_KC=['biochemistry','genetics','molecular biology','recombinant DNA technology','genomics','rational medicine','building blocks of life','interactions dictate structure','function in biology','predict genotypes and phenotypes','genetics data','central dogma of molecular biology',' convert DNA sequence to RNA sequence to protein sequence',' molecular tools to study biology',' principles of modern biology to issues in todays society']
course12_KC=['Newtonian Mechanics ','force','moves on to straight-line motion','momentum','mechanical energy','rotational motion','angular momentum','harmonic oscillators','planetary orbits','structure of Mechanics','conservation laws']


course_all=course1_KC+course2_KC+course3_KC+course4_KC+course5_KC+course6_KC+course7_KC+course8_KC+course9_KC+course10_KC+course11_KC+course12_KC
mylist=list(set(course_all))
print(" A few of edX courses are : \n")
for i,row in enumerate(mylist):
    print(i+1," ",row)

print("\n")

topic=input("Enter the subject/topic of your interets : ")
yob=input("\nEnter your year of birth : ")
education_level=input("Enter your highest education level : ")
country=input("Enter your country : ")
gender=input("Enter your gender : ")


courses_names=['Ancient Greek Hero','Intro to CS','Justice','Health Statictics','Health Environment','Poverty','Electronics and Magnetism','Circuits and Electronics','Solid State Chemistry','Intro to CS and Programming','Biology','Mechanics Review']
print("\n")
if(topic in course1_KC):
    print(courses_names[0],"\n COURSE DESCRIPTION : This is a course about Greek culture, their language.\n")
    print("")
    certified(courses_names[0])
    dropouts(courses_names[0])
    grade_dist(courses_names[0])
if(topic in course2_KC):
    print(courses_names[1],"\n COURSE DESCRIPTION : This is a cousrse giving basic introduction to computer science \n")
    print("")
    certified(courses_names[1])
    dropouts(courses_names[1])
    grade_dist(courses_names[1])
if(topic in course3_KC):
    print(courses_names[2],"\n COURSE DESCRIPTION : This is a course of Law and Justice \n")
    print("")
    certified(courses_names[2])
    dropouts(courses_names[2])
    grade_dist(courses_names[2])
if(topic in course4_KC):
    print(courses_names[3]," \n COURSE DESCRIPTION : This is a course on statistics in Health Domain, which is turning out to be an important in Data Science area\n")
    print("")
    certified(courses_names[3])
    dropouts(courses_names[3])
    grade_dist(courses_names[3])
if(topic in course5_KC):
    print(courses_names[4],"\n COURSE DESCRIPTION : This is a course on Health Environment\n")
    print("")
    certified(courses_names[4])
    dropouts(courses_names[4])
    grade_dist(courses_names[4])
if(topic in course6_KC):
    print(courses_names[5],"\n COURSE DESCRIPTION : This is a course related to poverty in present day and it's effects \n")
    print("")
    certified(courses_names[5])
    dropouts(courses_names[5])
    grade_dist(courses_names[5])
if(topic in course7_KC):
    print(courses_names[6],"\n COURSE DESCRIPTION : This is a course of how magnetism and electronics are interrelated\n")
    print("")
    certified(courses_names[6])
    dropouts(courses_names[6])
    grade_dist(courses_names[6])
if(topic in course8_KC):
    print(courses_names[7],"\n COURSE DESCRIPTION : This is a Electronics course, digged deep into circuits\n")
    print("")
    certified(courses_names[7])
    dropouts(courses_names[7])
    grade_dist(courses_names[7])
if(topic in course9_KC):
    print(courses_names[8],"\n COURSE DESCRIPTION : This is a Science course, concentrated on Solid states\n")
    print("")
    certified(courses_names[8])
    dropouts(courses_names[8])
    grade_dist(courses_names[8])
if(topic in course10_KC):
    print(courses_names[9],"\n COURSE DESCRIPTION : This is a an intro/basic course of CS - concentrated on Programming\n")
    print("")
    certified(courses_names[9])
    dropouts(courses_names[9])
    grade_dist(courses_names[9])
if(topic in course11_KC):
    print(courses_names[10],"\n COURSE DESCRIPTION : This is a Science course concentrated on Biology\n")
    print("")
    certified(courses_names[10])
    dropouts(courses_names[10])
    grade_dist(courses_names[10])
if(topic in course12_KC):
    print(courses_names[11],"\nCOURSE DESCRIPTION : This is a Physics course concentrated on Mechanics \n")
    print("")
    certified(courses_names[11])
    dropouts(courses_names[11])
    grade_dist(courses_names[11])

predict_course(yob,education_level,country,gender)

