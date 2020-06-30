#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[18]:


black_friday.head()


# In[19]:


black_friday.isna()


# In[20]:


int(black_friday['Product_Category_3'].mode())


# In[21]:


# normalização
def norm(df, column):
    aux = df[column]
    norm_aux = (aux - aux.min())/(aux.max() - aux.min())

    return norm_aux


# In[22]:


norm_purch = norm(black_friday, 'Purchase')


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[23]:


def q1():
    return black_friday.shape
    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[24]:


def q2():
    filter = 'Gender == "F" and Age == "26-35"'
    count = black_friday.query(filter).shape[0]
    return count

    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[25]:


def q3():
    return black_friday['User_ID'].unique().shape[0]


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[26]:


def q4():
    return black_friday.dtypes.unique().shape[0]


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[27]:


def q5():
    total_samples = black_friday.shape[0]
    nb_nan = total_samples - black_friday.dropna().shape[0]
    
    return nb_nan/total_samples


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[28]:


def q6():
    return int(black_friday.isna().sum().sort_values(ascending=False)[0])


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[29]:


def q7():
    return int(black_friday['Product_Category_3'].mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[30]:


def q8():
    return float(norm(black_friday, 'Purchase').mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[53]:


def q9():
    purch = black_friday['Purchase']
    normalized = (purch - purch.mean()) / purch.std()
    return (normalized.between(-1, 1).sum())
    
    


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[36]:


def q10():
    pass
    

