import pandas as pd
import os
import numpy as np

def calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores, coluna_cognitiva):
    """
    Calcula a média dos valores de uma coluna cognitiva para os jogadores especificados.

    Parâmetros:
    - data_rodada: pd.Series com os dados da rodada (contendo os números dos jogadores)
    - resultados_cog: pd.DataFrame com os dados cognitivos
    - indices_jogadores: lista de inteiros indicando as posições dos jogadores na rodada
    - coluna_cognitiva: string com o nome da coluna cognitiva desejada

    Retorna:
    - média dos valores encontrados ou np.nan se nenhum valor for encontrado
    """
    valores = []
    for i in indices_jogadores:
        jogador_id = data_rodada.iloc[i]
        filtro = resultados_cog.loc[resultados_cog['Numero'] == jogador_id, coluna_cognitiva]
        if not filtro.empty:
            valores.append(filtro.values[0])
    
    return np.mean(valores) if valores else np.nan


def analisar_rodada(cog_coletivo, resultados_cog, data_rodada, campo, rodada):
    
    #gol feitos por cada time
    gols_time_1 = data_rodada.iloc[1] + data_rodada.iloc[3] + data_rodada.iloc[5]
    gols_time_2 = data_rodada.iloc[7] + data_rodada.iloc[9] + data_rodada.iloc[11]
    
    #Saldo de gols time
    saldo_gols_time_1 = gols_time_1-gols_time_2
    
    #anotar resultado
    if saldo_gols_time_1>0:
        
        cog_coletivo.loc[f'{rodada+1}_{campo+1}','resultado'] = 1
        
    else:
        
        cog_coletivo.loc[f'{rodada+1}_{campo+1}','resultado'] = 0
        
    #Atenção sustentada Time 1  
    media_as_1 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[0, 2, 4], coluna_cognitiva='Acuracia Go')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'AS_1'] = media_as_1
   
    #Atenção sustentada Time 2
    media_as_2 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[6, 8, 10], coluna_cognitiva='Acuracia Go')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'AS_2'] = media_as_2

    #Memória de Trabalho Time 1  
    media_mtv_1 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[0, 2, 4], coluna_cognitiva='Memory span')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'MTV_1'] = media_mtv_1

    #Memória de Trabalho Time 2  
    media_mtv_2 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[6, 8, 10], coluna_cognitiva='Memory span')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'MTV_2'] = media_mtv_2
    
    #Flexibilidade cognitiva Time 1  
    media_fc_1 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[0, 2, 4], coluna_cognitiva='Flexibilidade cognitiva (B-A)')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'FC_1'] = media_fc_1
    
    #Flexibilidade cognitiva Time 2  
    media_fc_2 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[6, 8, 10], coluna_cognitiva='Flexibilidade cognitiva (B-A)')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'FC_2'] = media_fc_2
    
    #Impulsividade Time 1  
    media_i_1 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[0, 2, 4], coluna_cognitiva='Acuracia nogo')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'I_1'] = media_i_1
    
    #Impulsividade Time 2  
    media_i_2 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[6, 8, 10], coluna_cognitiva='Acuracia nogo')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'I_2'] = media_i_2
    
    #Capacidade de rastreamento Time 1  
    media_cr_1 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[0, 2, 4], coluna_cognitiva='Capacidade de rastreamento')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'CR_1'] = media_cr_1
    
    #Capacidade de rastreamento Time 2  
    media_cr_2 = calcular_media_cognitiva(data_rodada, resultados_cog, indices_jogadores=[6, 8, 10], coluna_cognitiva='Capacidade de rastreamento')
    cog_coletivo.loc[f'{rodada+1}_{campo+1}', 'CR_2'] = media_cr_2
    
    return cog_coletivo
    
    
def calcVariaveis(resultados, resultados_cog, campos, rodadas):
    
    # Criando os índices (nomes das linhas)
    indices = []
    for rodada in range(1, rodadas + 1):
        for campo in range(1, campos + 1):
            indices.append(f'{rodada}_{campo}')
            
    # Criando o DataFrame vazio com esses índices
    cog_coletivo = pd.DataFrame(index=indices)

    cog_coletivo[['AS_1', 'MTV_1', 'FC_1', 'I_1', 'CR_1', 'PCT_1', 'AS_2', 'MTV_2', 'FC_2', 'I_2', 'CR_2', 'PCT_2', 'resultado']] = 0
    
    #ajustando resultados
    resultados = resultados.iloc[2:, 3:]
    
    #pecorre todos os campos
    for campo_id in range(campos):  # campo_id = 0,1,2,3
        campo = campo_id * 12  # índice real de coluna
        
        #Percorre todas as rodadas
        for rodada in range(0, rodadas):
            
            #seleciona os dados da rodada
            data_rodada = resultados.iloc[rodada, campo:campo+12]
            
            #extrai as variaveis dos dados da rodada
            cog_coletivo = analisar_rodada(cog_coletivo, resultados_cog, data_rodada, campo_id, rodada)    
            
    return cog_coletivo
    

def media_jogadores(resultados, resultados_cog, num_jog = 20):
    

    #Calcular gols feitos, sofridos, feitos por companheiros e saldo final
    cog_coletivo = calcVariaveis(resultados, resultados_cog, campos = 4, rodadas = 28)
    
    
    return cog_coletivo
    
def save_csv(path_to_save, performance_jogadores, performance_jogadores_bruto):
    path_performance_jogadores = os.path.join(path_to_save, 'Media_performance_jogadores.csv')
    path_performance_jogadores_bruto = os.path.join(path_to_save, 'Total_performance_jogadores.csv')
    
    performance_jogadores.to_csv(path_performance_jogadores)
    performance_jogadores_bruto.to_csv(path_performance_jogadores_bruto)
    
if __name__ == "__main__":
    
    path_to_data_reduzidos = 'D:\\Collective_cog_prediction_MLP\\data\\Resultado_jogos_time_2.xlsx'
    path_to_data_cog = 'D:\\Collective_cog_prediction_MLP\\data\\full_cog_data_time_2.csv'
    
    # Ler o arquivo Excel
    resultados_reduzidos = pd.read_excel(path_to_data_reduzidos)
    resultados_cog = pd.read_csv(path_to_data_cog)
    
    #ver médida de cada jogador
    cog_coletivo = media_jogadores(resultados_reduzidos, resultados_cog, num_jog = 30)
        
    #Salvar arquivo
    cog_coletivo.to_csv('D:\\Collective_cog_prediction_MLP\\data\\pre_processado\\cog_coletivo_time_2.csv')