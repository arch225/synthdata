import streamlit as st
import pandas as pd
import pickle
import sqlite3
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer,GaussianCopulaSynthesizer,TVAESynthesizer
from sdv.lite import SingleTablePreset


def Gen_data(model,rows):
    try:
        if model=='FastML':
            m_obj='my_synth_fml_v1.pkl'
        elif model=='Gaussian Copula Synthesizer':
            m_obj='my_synth_gcs_v1.pkl'
        elif model=='CT-GAN':
            m_obj='my_synth_ctg_v1.pkl'
        elif model=='TVAE':
            m_obj='my_synth_tvae_v1.pkl'
        with open(m_obj,'rb') as f:
            
            syn=pickle.load(f)
            row=int(rows)
            
            syn_gen_data=syn.sample(num_rows=row)
            return st.write(syn_gen_data)
    except ValueError:
        return "Invalid option chosen."
    
def db_tables(db):
    conn = sqlite3.connect(db)
    sql_query = """SELECT name FROM sqlite_master  
    WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    df_address=pd.read_sql_query("SELECT * FROM ing_address",conn)
    df_customer=pd.read_sql_query("SELECT * FROM ing_customer",conn)
    df_demographic=pd.read_sql_query("SELECT * FROM ing_demographic",conn)
    df_termination=pd.read_sql_query("SELECT * FROM ing_termination",conn)
    df_mer1=pd.merge(df_customer,df_address,on='ADDRESS_ID',how='left')
    df_mer2=pd.merge(df_mer1,df_demographic,on='INDIVIDUAL_ID',how='left')
    df_mer3=pd.merge(df_mer2,df_termination,on='INDIVIDUAL_ID',how='left')
    st.write(df_mer3.head(10))
    return df_mer3


def impute_all():
    st.write('Starting Imputation..')
    df2=df_mer3[df_mer3['INDIVIDUAL_ID'].notna()]
    df2['ACCT_SUSPD_DATE']=np.where(df2['ACCT_SUSPD_DATE'].isnull(),'2025-01-01',df2['ACCT_SUSPD_DATE'])
    d=[]
    for i in df2.columns:
        if df2[i].isna().sum()>0:
            d.append(i)
    else:
        print("Suthara hai bhenchod: ",i,"\s",df2[i].isna().sum())
    for i in d:
        if (df2[i].dtype == 'int64') or (df2[i].dtype == 'float64'):
            df2[i]=df2[i].fillna(df2[i].median())
        elif df2[i].dtype == 'object':
            df2[i]=df2[i].fillna(df2[i].mode()[0])
    st.write("Imputation status",df2.isna().sum())
    return df2

def meta_validation(dat):
    st.write('Validating Metadata...')
    metadata=SingleTableMetadata()
    metadata.detect_from_dataframe(dat)
    py_dict=metadata.to_dict()
    st.write(py_dict)
    st.write('Validation completed')
    return metadata

def train_fml(dat,metadata,out_pkl):
    st.write('Training starting...')
    synth = SingleTablePreset(metadata, name='FAST_ML')
    synth.fit(df2)
    synth.save(filepath=out_pkl)
    st.write(f'Training completed, model is saved: {out_pkl}')

def train_gcs(dat,metadata,out_pkl):
    st.write('Training starting...')
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df2)
    synth.save(filepath=out_pkl)
    st.write(f'Training completed, model is saved: {out_pkl}')

def train_ctgan(dat,metadata,out_pkl):
    st.write('Training starting...')
    synth = CTGANSynthesizer(metadata,epochs=100,verbose=True)
    synth.fit(df2)
    synth.save(filepath=out_pkl)
    st.write(f'Training completed, model is saved: {out_pkl}')

def train_tvae(dat,metadata,out_pkl):
    st.write('Training starting...')
    synth = TVAESynthesizer(metadata,epochs=100,verbose=True)
    synth.fit(df2)
    synth.save(filepath=out_pkl)
    st.write(f'Training completed, model is saved: {out_pkl}')


st.title('Welcome to V1 version of data synthesizer :smile:')
task=st.sidebar.selectbox('Select your choice:',['Train a model','Generate data'])            

st.divider()

if task=='Train a model':
    st.write('Connect to data source: Database or Flatfiles')
    dat_src=st.radio('Choose data source',['Connect to a db','Upload files'])
    st.divider()
    if dat_src=='Connect to a db':
        st.write('Connecting to db...')
        df_mer3=db_tables('Source_data.db')
        st.write('Data has been loaded and combined')
        st.divider()
        st.write('Missing value status: ',df_mer3.isna().sum())
        st.write('Imputing all')
        df2=impute_all()
        st.write('Imputation completed')
        st.divider()
        metadata=meta_validation(df2)
        m_choice=['FastML','Gaussian Copula Synthesizer','CT-GAN','TVAE']
        mt_selection=st.selectbox('Choose a model to generate data: ',m_choice,index=0)
        epoch=st.number_input('Mention number of epochs',min_value=1,max_value=1000)
        if mt_selection=='FastML':
            train_fml(df2,metadata,'custom_model/my_synth_fml_sys_v1.pkl')
        elif mt_selection=='Gaussian Copula Synthesizer':
            train_gcs(df2,metadata,'custom_model/my_synth_gcs_sys_v1.pkl')
        elif mt_selection=='CT-GAN':
            train_ctgan(df2,metadata,'custom_model/my_synth_ctgan_sys_v1.pkl')
        elif mt_selection=='TVAE':
            train_tvae(df2,metadata,'custom_model/my_synth_tvae_sys_v1.pkl')
        
    elif dat_src=='Upload files':
        st.file_uploader('Upload file')
elif task=='Generate data':
    models=['FastML','Gaussian Copula Synthesizer','CT-GAN','TVAE','Custom model']
    m_selection=st.selectbox('Choose a model to generate data: ',models,index=0)
    rows=st.number_input('Number of rows to be generated: ',min_value=1,max_value=1000000,value=1000)
    st.write(f'So you have chosen model: {m_selection} and {rows} number of rows to be generated')
    gen_button=st.button('Generate Data')
    st.divider()
    if gen_button:
        st.write('Generating data....Please wait..')
        df_synth=Gen_data(m_selection,rows)
        st.write('Disintegrating tables..')
        

        df_temp=df_synth['ADDRESS_ID', 'LATITUDE', 'LONGITUDE', 'STREET_ADDRESS', 'CITY','STATE', 'COUNTY']
        df_address=df_temp.drop_duplicates(subset=['ADDRESS_ID'],keep='last')
        df_temp=df_synth['INDIVIDUAL_ID', 'ADDRESS_ID', 'CURR_ANN_AMT', 'DAYS_TENURE','CUST_ORIG_DATE', 'AGE_IN_YEARS', 'DATE_OF_BIRTH','SOCIAL_SECURITY_NUMBER']
        df_customer=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')
        df_temp=df_synth['INDIVIDUAL_ID', 'INCOME', 'HAS_CHILDREN', 'LENGTH_OF_RESIDENCE','MARITAL_STATUS', 'HOME_MARKET_VALUE', 'HOME_OWNER', 'COLLEGE_DEGREE','GOOD_CREDIT']
        df_demographic=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')
        df_temp=df_synth['INDIVIDUAL_ID', 'ACCT_SUSPD_DATE']
        df_termination=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')
        one,two,three,four=st.columns(4)
        with one:
            st.write(df_address)
        with two:
            st.write(df_customer)
        with three:
            st.write(df_demographic)
        with four:
            st.write(df_termination)
        

