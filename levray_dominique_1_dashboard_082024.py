'''
========================================================================
Projet n°8 - Réalisez un dashboard et assurez une veille technique
Script Python réalisé par Dominique LEVRAY en Août/Septembre 2024
========================================================================
Note : Ce dashboard est la version 2 de l'IHM developpée pour le projet
       n°7 (page streamlit de test de l'API déployée sur HEROKU)

Ce fichier répond aux demandes suivantes :

- Permettre de visualiser le score, sa probabilité (est-il loin du seuil ?)
  et l’interprétation de ce score pour chaque client de façon intelligible
  pour une personne non experte en data science.

- Permettre de visualiser les principales informations descriptives relatives
  à un client.

- Permettre de comparer, à l’aide de graphiques, les principales informations
  descriptives relatives à un client à l’ensemble des clients ou à un groupe
  de clients similaires (via un système de filtre : par exemple,
  liste déroulante des principales variables).

Pour exécuter ce fichier : streamlit run levray_dominique_1_dashboard_082024.py
'''
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=broad-exception-raised
# pylint: disable=trailing-whitespace

# -------------------------------------------------------------------------------------------------------------------
# Importation des modules
# -------------------------------------------------------------------------------------------------------------------

from    scipy.ndimage       import gaussian_filter1d

import  json
import  requests
import  numpy               as np
import  pandas              as pd
import  streamlit           as st
import  matplotlib          as pl
import  matplotlib.pyplot   as plt
import  shap
import  pickle
import  zipfile

# -------------------------------------------------------------------------------------------------------------------
# Définition des constantes
# -------------------------------------------------------------------------------------------------------------------

# Echange avec l'API
API_BASE_URL  = "https://oc-projet-7-c21cbfffa8fb.herokuapp.com"    # URL de base de l'API sur heroku
# API_BASE_URL  = "http://127.0.0.1:8000"                             # URL de base de l'API en local
API_MATRICE   = "/matrice_confusion"
API_GET       = "/get_client/"
API_POST      = "/post_client/{Client_new_credit}"
API_PRED      = "/prediction/{data}"
HTTPS_TIMEOUT = 30                                                  # Timeout pour les requettes https (30 secondes)

BEST_THRESHOLD = 0.27                                               # Seuil de probabilité pour la fixation de la TARGET

IS_SUBMITTED   = "IS_SUBMITTED"      # Flag permettant de savoir si une demande de nouvelle prédiction a été faite (sinon => valeurs initiales)

# Couleurs contratées compatible WCAG
CONSTRASTED_COLOR1 = "#6F0000"
CONSTRASTED_COLOR2 = "#000860"

# Identifiants des widgets
SELECT_CLIENT           = "SELECT_CLIENT"           # selectbox de sélection du client
VARIABLE_X              = "VARIABLE_X"              # selectbox de sélection de la variable à placer en X sur le graphique de corrélation
VARIABLE_Y              = "VARIABLE_Y"              # selectbox de sélection de la variable à placer en Y sur le graphique de corrélation
SLIDER_AMT_CREDIT       = "SLIDER_AMT_CREDIT"       # double slider de sélection de la fourchette de valeur pour AMT_CREDIT
SLIDER_AGE              = "SLIDER_AGE"              # double slider de sélection de la fourchette de valeur pour age
SLIDER_AMT_GOODS_PRICE  = "SLIDER_AMT_GOODS_PRICE"  # double slider de sélection de la fourchette de valeur pour AMT_GOODS_PRICE
SLIDER_ANCIENNETE       = "SLIDER_ANCIENNETE"       # double slider de sélection de la fourchette de valeur pour anciennete
SLIDER_EXT_SOURCES      = "SLIDER_EXT_SOURCES"      # double slider de sélection de la fourchette de valeur pour SLIDER_EXT_SOURCES 1 à 3

CORRELATION_POSSIBLE_VAR = ["age", "anciennete", "AMT_GOODS_PRICE", "AMT_CREDIT"]

# Nom de fichier
ZIP_TEST_DATA_FILENAME      = "test_data.zip"                       # fichier contenant les données de test
ZIP_PREPARED_DATA_FILENAME  = "prep_data.zip"                       # fichier contenant les données préparées (pour affichage de graphiques et KPI)
ZIP_SHAP_EXPLAINER          = "tree_explainer.zip"                                      # fichier zip contenant l'explainer SHAP
INSIDE_ZIP_SHAP_EXPLAINER   = "tree_explainer.pkl"                                      # fichier à l'intérieur du zip contenant l'explainer SHAP
SHAPVALUES_DATA_FILENAME    = "shap_data.pkl"                                           # fichier contenant les données "shap values"

# Des styles CSS pour la mise en forme
STYLE_KPI_BIG      = "style=font-size:14px;font-weight:700;"
STYLE_SMALL_ITALIC = "style=font-size:10px;font-style:italic;"

# Liste de colonnes "onehotencoded" à regrouper
# Note :
#   - Les versions LIST servent aux cammenbert ou contiennent toutes les valeurs possibles
#   - Les versions LISTFULL contiennent toutes les valeurs possibles quand une version cammenbert existe

NAME_EDUCATION_TYPE_LIST        = ["NAME_EDUCATION_TYPE_Secondarysecondaryspecial", "NAME_EDUCATION_TYPE_Highereducation"]
NAME_EDUCATION_TYPE_LISTFULL    = NAME_EDUCATION_TYPE_LIST+["NAME_EDUCATION_TYPE_Incompletehigher", "NAME_EDUCATION_TYPE_Lowersecondary"]
NAME_INCOME_TYPE_LIST           = ['NAME_INCOME_TYPE_Working', 'NAME_INCOME_TYPE_Commercialassociate', 'NAME_INCOME_TYPE_Stateservant']
NAME_FAMILY_STATUS_LIST         = ['NAME_FAMILY_STATUS_Civilmarriage', 'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Singlenotmarried']
NAME_FAMILY_STATUS_LISTFULL     = NAME_FAMILY_STATUS_LIST+['NAME_FAMILY_STATUS_Widow']
WALLSMATERIAL_MODE_LIST         = ['WALLSMATERIAL_MODE_Panel', 'WALLSMATERIAL_MODE_Stonebrick']
WALLSMATERIAL_MODE_LISTFULL     = WALLSMATERIAL_MODE_LIST+['WALLSMATERIAL_MODE_Block']
NAME_TYPE_SUITE_LIST            = ['NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Unaccompanied']
OCCUPATION_TYPE_LIST            = ['OCCUPATION_TYPE_Corestaff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_LowskillLaborers',
                                   'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Salesstaff', 'OCCUPATION_TYPE_Securitystaff']
ORGANIZATION_TYPE_LIST          = ['ORGANIZATION_TYPE_BusinessEntityType3', 'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Selfemployed']
WEEKDAY_APPR_PROCESS_START_LIST = ['WEEKDAY_APPR_PROCESS_START_FRIDAY', 'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY',
                                   'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY']
ONEHOTENCODED_LIST              = NAME_EDUCATION_TYPE_LISTFULL + NAME_INCOME_TYPE_LIST + NAME_FAMILY_STATUS_LISTFULL + WALLSMATERIAL_MODE_LISTFULL + \
                                  NAME_TYPE_SUITE_LIST + OCCUPATION_TYPE_LIST + ORGANIZATION_TYPE_LIST + WEEKDAY_APPR_PROCESS_START_LIST


# -------------------------------------------------------------------------------------------------------------------
# Appel l'API pour faire une prédiction
# -------------------------------------------------------------------------------------------------------------------

def get_prediction(data_df):
    '''Appel l'API pour faire une prédiction à partir des données présentes dans data_df
       En entrée : data_df ne doit contenir qu'une seule ligne
       En sortie : la probabilité et la target
    '''

    def data_row_to_dict(data_row_df):
        '''Construit un dictionnaire à partir de la première ligne d'un dataframe'''
        col_types = data_row_df.dtypes.to_list()
        col_names = data_row_df.columns.to_list()
        ret_dict  = {}
        for col_num in range(0, len(col_names)):
            key           = f"{col_types[col_num]}={col_names[col_num]}"
            if col_types[col_num] == "bool":                                            # bool n'est pas reconnu par json
                value = ('True' if data_row_df.iloc[0, col_num] else 'False')
            elif (col_types[col_num] == "int64") or (col_types[col_num] == "float64"):  # Passe en string pour ne pas perdre en précision
                value = str(data_row_df.iloc[0, col_num])
            else:
                value = data_row_df.iloc[0, col_num]
            ret_dict[key] = value
        return ret_dict

    # Envoi une requete au serveur (Fourni les données sous forme d'un dictionnaire)
    result = requests.post(f"{API_BASE_URL}{API_PRED}", json=data_row_to_dict(data_df), timeout=HTTPS_TIMEOUT)

    # Decode et convertie le JSON format en dictionaire
    dict_obj = json.loads(result.content)

    # Décodage du client
    y_proba = float(dict_obj.get("y_proba"))
    y_pred  = int(dict_obj.get("y_pred"))

    return y_proba, y_pred


# -------------------------------------------------------------------------------------------------------------------
# Pré-traitement des données
# -------------------------------------------------------------------------------------------------------------------

def data_transform(unfiltered_data):
    '''Fonction de calculs des données avant visualisation'''

    # Initialise et/ou récupère les filtres sur les valeurs quantitatives
    # Par défaut : prend les min et max

    def prep_get_minmax(widget, feature):
        if widget not in st.session_state:
            min_v = prepared_data_df[feature].min()
            max_v = prepared_data_df[feature].max()
        else:
            min_v = st.session_state[widget][0]
            max_v = st.session_state[widget][1]
        return min_v, max_v

    min_AMT_CREDIT, max_AMT_CREDIT           = prep_get_minmax(SLIDER_AMT_CREDIT, 'AMT_CREDIT')
    min_age, max_age                         = prep_get_minmax(SLIDER_AGE, 'age')
    min_AMT_GOODS_PRICE, max_AMT_GOODS_PRICE = prep_get_minmax(SLIDER_AMT_GOODS_PRICE, 'AMT_GOODS_PRICE')
    min_anciennete, max_anciennete           = prep_get_minmax(SLIDER_ANCIENNETE, 'anciennete')
    min_EXT_SOURCES, max_EXT_SOURCES         = prep_get_minmax(SLIDER_EXT_SOURCES, 'EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3')

    # Applique les filtres sur les données
    filtered_data = unfiltered_data[(unfiltered_data["AMT_CREDIT"] >= min_AMT_CREDIT) & (unfiltered_data["AMT_CREDIT"] <= max_AMT_CREDIT) &
                                    (unfiltered_data["age"] >= min_age) & (unfiltered_data["age"] <= max_age) &
                                    (unfiltered_data["AMT_GOODS_PRICE"] >= min_AMT_GOODS_PRICE) & (unfiltered_data["AMT_GOODS_PRICE"] <= max_AMT_GOODS_PRICE) &
                                    (unfiltered_data["anciennete"] >= min_anciennete) & (unfiltered_data["anciennete"] <= max_anciennete) &
                                    (unfiltered_data["EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3"] >= min_EXT_SOURCES) & (unfiltered_data["EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3"] <= max_EXT_SOURCES)]

    # Constantes utilisées pour faire les calculs
    BOOLEAN_FEATURES = ['NAME_EDUCATION_TYPE_Secondarysecondaryspecial',
                        'NAME_EDUCATION_TYPE_Highereducation',

                        'FLAG_OWN_CAR',

                        'NAME_FAMILY_STATUS_Civilmarriage',
                        'NAME_FAMILY_STATUS_Married',
                        'NAME_FAMILY_STATUS_Separated',
                        'NAME_FAMILY_STATUS_Singlenotmarried',

                        'WALLSMATERIAL_MODE_Panel',
                        'WALLSMATERIAL_MODE_Stonebrick',

                        'NAME_INCOME_TYPE_Working',
                        'NAME_INCOME_TYPE_Commercialassociate',
                        'NAME_INCOME_TYPE_Pensioner',
                        'NAME_INCOME_TYPE_Stateservant']

    KPI_FEATURES = ['AMT_CREDIT',
                    'AMT_GOODS_PRICE',
                    'EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3',
                    'OBS_30_CNT_SOCIAL_CIRCLE',
                    'OWN_CAR_AGE']

    def prep_distrib(feat, featname):
        names   = np.linspace(start=filtered_data[feat].min(), stop=filtered_data[feat].max(), num=25, endpoint=True)
        bar_val = np.histogram(filtered_data[feat], bins=25, density=False)[0]
        target0 = np.histogram(filtered_data.loc[filtered_data['TARGET'] == 0, feat], bins=25, density=True)[0]
        target1 = np.histogram(filtered_data.loc[filtered_data['TARGET'] == 1, feat], bins=25, density=True)[0]
        return pd.DataFrame({featname+"_x": names, featname+"_distribution": bar_val, featname+"_target_0": target0, featname+"_target_1": target1})

    # -------------------------------------------------------------------------
    # Prépare les valeurs qui seront utilisées pour afficher les distributions
    # -------------------------------------------------------------------------

    distributions_df = prep_distrib("age", "age")                                                                 # Prépare les données pour le graphique "Effet de l'âge sur le remboursement"
    distributions_df = pd.concat([distributions_df, prep_distrib("anciennete",      "anciennete")], axis=1)       # Prépare les données pour le graphique "Effet de l'anciennete sur le remboursement"
    distributions_df = pd.concat([distributions_df, prep_distrib("AMT_CREDIT",      "AMT_CREDIT")], axis=1)       # Prépare les données pour le graphique "Effet du montant du crédit sur le remboursement"
    distributions_df = pd.concat([distributions_df, prep_distrib("AMT_GOODS_PRICE", "AMT_GOODS_PRICE")], axis=1)  # Prépare les données pour le graphique "Effet du prix du bien sur le remboursement"
    distributions_df = pd.concat([distributions_df, prep_distrib("EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3", "EXT_SOURCES")], axis=1)  # Prépare les données pour le graphique "Effet de la variable EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3 sur le remboursement"

    # -------------------------------------------------------------------------
    # Prépare les valeurs qui seront utilisées pour afficher les camenberts
    # -------------------------------------------------------------------------
    boolean_number = []
    for feat in BOOLEAN_FEATURES:
        boolean_number.append(filtered_data[filtered_data[feat]==True].shape[0])

    BOOLEAN_FEATURES.append('Total')
    boolean_number.append(filtered_data.shape[0])

    boolean_df = pd.DataFrame(columns=BOOLEAN_FEATURES)   # Créer un dataframe vide
    boolean_df.loc[0] = boolean_number                    # Ajouter une ligne à partir de la liste

    # -------------------------------------------------------------------------
    # Prépare les valeurs qui seront utilisées pour afficher les KPI
    # -------------------------------------------------------------------------

    kpi_mean_values = []
    for feat in KPI_FEATURES:
        kpi_mean_values.append(filtered_data[feat].mean())

    kpi_df = pd.DataFrame(columns=KPI_FEATURES)   # Créer un dataframe vide
    kpi_df.loc[0] = kpi_mean_values               # Ajouter une ligne à partir de la liste

    return distributions_df, boolean_df, kpi_df


# -------------------------------------------------------------------------------------------------------------------
# Définition des graphiques
# -------------------------------------------------------------------------------------------------------------------

def draw_kpi(val_moyenne, val_client, texte):
    '''Affiche un KPI (moyenne) pour une variable quantitative'''

    text_moyenne = (f"{val_moyenne/1000:.1f}K"        if (val_moyenne > 10000) else f"{val_moyenne:.1f}")
    text_client  = (f"Client={val_client/1000:.1f}K"  if (val_client  > 10000) else f"Client={val_client:.1f}")

    zone_kpi = st.container(border=True)
    # Instancie une figure matplotlib
    fig = plt.figure(figsize = (2, 1))
    ax = fig.add_subplot(111, xlim=(0, 1), ylim=(0, 1), autoscale_on=False, clip_on=True)
    ax.set_axis_off()
    ax.text(0.5, 0.9, texte,        horizontalalignment='center', verticalalignment='center', fontsize=7,  fontweight='bold', clip_on=True)
    ax.text(0.5, 0.4, text_moyenne, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold', clip_on=True)
    ax.text(0.5, 0.1, text_client,  horizontalalignment='center', verticalalignment='center', fontsize=7,  fontweight='bold', clip_on=True)
    zone_kpi.pyplot(fig)
    plt.close()


# --------------------------------------------------------

def fmt_BMK(tick_val, pos):
    '''
    Fonction utilitaire pour changer le format d'un axe
    Affiche des K au dela de 10K
    Affiche des Mds au dela de 1 million
    Affiche des Mrd au dela de 100 millions
    '''
    if tick_val >= 100_0000_000:
        return f"{(tick_val/1_000_000_000):.1f} Mrd"
    elif tick_val >= 1_000_000:
        return f"{(tick_val/1_000_000):.1f} Mds"
    elif tick_val >= 10_000:
        return f"{(tick_val/1_000):.1f} K"
    else:
        return '{:.2f}'.format(tick_val)


# --------------------------------------------------------

def graph_bivariee(distributions_df):
    ''' Affichage d'un graphique de dispersion (scatter plot) pour représenter une corrélation entre 2 features'''

    # Récupère les variables sélectionnées par l'utilisateur
    feature1 = st.session_state[VARIABLE_X]
    feature2 = st.session_state[VARIABLE_Y]

    zone_graph = st.container(border=False)

    fig = plt.figure(figsize = (6, 4))
    ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1, rowspan=1)

    # Affichage d'un nuage de points
    ax1.scatter(distributions_df[feature1], distributions_df[feature2])

    # Affichage du titre et des labels
    plt.title(f"Analyse bi-variée entre {feature1} et {feature2}")
    ax1.set_xlabel(feature1, fontsize=12)
    ax1.set_ylabel(feature2, fontsize=12)
    ax1.tick_params(axis='both', labelsize=8)

    # Améliore l'affichage des axes
    _ = ax1.xaxis.set_major_formatter(pl.ticker.FuncFormatter(fmt_BMK))
    _ = ax1.yaxis.set_major_formatter(pl.ticker.FuncFormatter(fmt_BMK))

    # Active la grille
    ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    zone_graph.pyplot(fig)
    plt.close()


# --------------------------------------------------------

def graph_distrib(distributions_df, feature, valeur_client: int = 0, xlabel="", title="", use_fmt_BMK=False):
    '''Affiche un graphique de distribution pour une variable quantitative'''

    zone_graph = st.container(border=False)

    # Instancie une figure matplotlib et un subplot
    fig = plt.figure(figsize = (6, 4))
    ax1 = plt.subplot2grid(shape=(1, 1), loc=(0, 0), colspan=1, rowspan=1)

    # Affiche la distribution
    width = (distributions_df[feature+"_x"].max()-distributions_df[feature+"_x"].min())/25
    ax1.bar(distributions_df[feature+"_x"], height=distributions_df[feature+"_distribution"], align='edge', edgecolor = 'k', color='lightblue', width=width)

    # Instantie un deuxième axe Y qui partage le même axe X
    ax2 = ax1.twinx()

    # Affiche la courbes représentant le nombre de prêt non remboursés (adoucie la courbe)
    ax2.plot(distributions_df[feature+"_x"], gaussian_filter1d(distributions_df[feature+"_target_1"], sigma=1), color=CONSTRASTED_COLOR1, label='Prêts non remboursés')

    # Affiche un trait vertical pour montrer la position du client
    if valeur_client > 0:
        ax2.axvline(valeur_client, color = 'red', linewidth=3, linestyle='--', label="Valeur pour ce client")

    # Active l'affichage des legendes
    _ = ax2.legend(fontsize=8)

    # Change les couleurs du deuxième axe pour utiliser la même que pour la courbe
    ax2.yaxis.label.set_color(CONSTRASTED_COLOR1)
    ax2.tick_params(axis='y', colors=CONSTRASTED_COLOR1)

    # Désactive les notations scientifiques et les offsets
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax2.ticklabel_format(useOffset=False, style='plain')

    # Fixe les labels des axes verticaux
    _ = ax1.set_ylabel('Nombre de crédit', fontsize=12)
    _ = ax2.set_ylabel('Densité', fontsize=12)

    # Active la grille
    ax1.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    # Fixe la taille des polices utilisées pour les ticks
    ax1.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)

    # Fixe label et titre
    _ = ax1.set_xlabel(xlabel, fontsize=12)
    _ = ax1.set_title(title)

    # Met en place une fonction de formatage des montants (si demandé)
    if use_fmt_BMK:
        _ = ax1.xaxis.set_major_formatter(pl.ticker.FuncFormatter(fmt_BMK))
        _ = ax2.xaxis.set_major_formatter(pl.ticker.FuncFormatter(fmt_BMK))

    text_client  = (f"Client={valeur_client/1000:.1f}K"  if (valeur_client  > 10000) else f"Client={valeur_client}")
    fig.text(0.5, -0.05, text_client,  horizontalalignment='center', verticalalignment='center', fontsize=10,  fontweight='bold', clip_on=False)

    zone_graph.pyplot(fig)
    plt.close()


# --------------------------------------------------------

def graph_boolean(data_df, start_feature, features, valeur_client="", title=""):
    '''Affiche un graphique sous forme de cammenbert à partir d'une liste de colonne onehotencoder'''

    zone_graph = st.container(border=True)

    # Compte les nombres d'occurence
    labels  = []
    valeurs = []
    explode = []
    cumul   = 0

    if len(features) > 1:
        for feat in features:
            part_feat = feat[len(start_feature)+1:]     # Coupe après le _
            nbr       = data_df[feat][0]
            cumul     += nbr
            labels.append(part_feat)
            explode.append((0.1 if feat == valeur_client else 0))
            valeurs.append(nbr)

        labels.append("Autre")
        valeurs.append(data_df['Total'][0]-cumul)
        explode.append((0.1 if 'Autre' == valeur_client else 0))
    else:
        nbr = data_df[features[0]][0]
        labels.append("Oui")
        explode.append((0.1 if valeur_client else 0))
        valeurs.append(nbr)
        labels.append("Non")
        valeurs.append(data_df['Total'][0]-nbr)
        explode.append((0.1 if not valeur_client else 0))

    # Instancie une figure matplotlib et un subplot
    fig = plt.figure(figsize = (6, 4))
    ax1 = plt.subplot2grid(shape=(3, 1), loc=(0, 0), rowspan=2, colspan=1)
    ax2 = plt.subplot2grid(shape=(3, 1), loc=(2, 0), rowspan=1, colspan=1)

    ax1.set_title(title)
    ax1.axis("equal")
    pie = ax1.pie(valeurs, explode=explode, autopct='%1.1f%%')
    plt.setp(pie[2], **{'color': 'white', 'weight': 'bold', 'fontsize': 8})     # pie[2] correspond aux pourcentages auto

    ax2.axis("off")
    ax2.legend(pie[0], labels, loc="center", fontsize=8)

    # Cas particulier pour les int
    if isinstance(valeur_client, np.int64):
        valeur_client = ("Oui" if valeur_client > 0 else "Non")
    elif valeur_client != "Autre":
        valeur_client = valeur_client[len(start_feature)+1:]

    fig.text(0.5, 0.05, "Client = "+valeur_client,  horizontalalignment='center', verticalalignment='center', fontsize=10,  fontweight='bold', clip_on=False)

    zone_graph.pyplot(fig)

    plt.close()


# --------------------------------------------------------

def get_var_from_col(col):
    '''En fonction du nom d'une colonne, détermine si elle appartient à une liste
        issue d'un onehotencoder et si oui, renvoi le nom initial (avant onehotencoder)
        et la liste de toutess les colonnes du groupe'''
    if col in NAME_EDUCATION_TYPE_LISTFULL:
        the_list  = NAME_EDUCATION_TYPE_LISTFULL
        start_col = "NAME_EDUCATION_TYPE"
    elif col in NAME_INCOME_TYPE_LIST:
        the_list  = NAME_INCOME_TYPE_LIST
        start_col = "NAME_INCOME_TYPE"
    elif col in NAME_FAMILY_STATUS_LISTFULL:
        the_list  = NAME_FAMILY_STATUS_LISTFULL
        start_col = "NAME_FAMILY_STATUS"
    elif col in WALLSMATERIAL_MODE_LISTFULL:
        the_list  = WALLSMATERIAL_MODE_LISTFULL
        start_col = "WALLSMATERIAL_MODE"
    elif col in NAME_TYPE_SUITE_LIST:
        the_list  = NAME_TYPE_SUITE_LIST
        start_col = "NAME_TYPE_SUITE"
    elif col in OCCUPATION_TYPE_LIST:
        the_list  = OCCUPATION_TYPE_LIST
        start_col = "OCCUPATION_TYPE"
    elif col in ORGANIZATION_TYPE_LIST:
        the_list  = ORGANIZATION_TYPE_LIST
        start_col = "ORGANIZATION_TYPE"
    elif col in WEEKDAY_APPR_PROCESS_START_LIST:
        the_list  = WEEKDAY_APPR_PROCESS_START_LIST
        start_col = "WEEKDAY_APPR_PROCESS_START"
    else:
        the_list  = []
        start_col = ""
    return the_list, start_col


# --------------------------------------------------------

def get_onehotencoded_value(current_client_df, list):
    '''Renvoie la valeur à True dans une liste de boolean encodé avec onehotencoder'''
    for col in list:
        if current_client_df[col][0]:
            return col
    return "Autre"


# --------------------------------------------------------

def get_current_client(test_data_df):
    '''En fonction de session_state, renseigne un dataframe d'une ligne avec le client en cours
        En entrée : le dataframe du jeu de données de test
        En sortie : un dataframe dans le même format que le jeu de test (compatible avec l'API)
    '''

    # print("dans get_current_client")
    # print(st.session_state)

    # Si c'est le premier lancement : Initialise le 1er ID client
    if SELECT_CLIENT not in st.session_state:

        # On choisi le 1er de la liste
        cur_id = test_data_df['SK_ID_CURR'].sort_values().reset_index(drop=True)[0]
        st.session_state[SELECT_CLIENT] = cur_id

        # Crée current client en l'initialisant depuis test_data avec le client sélectionné
        current_client_df = test_data_df[test_data_df['SK_ID_CURR'] == cur_id].copy().reset_index(drop=True)

    # Si le client n'a pas encore été changé (IS_SUBMITTED n'existe pas)
    elif IS_SUBMITTED not in st.session_state:

        # Récupère l'id depuis st.session_state
        cur_id = st.session_state[SELECT_CLIENT]

        # Crée current client en l'initialisant depuis test_data avec le client sélectionné
        current_client_df = test_data_df[test_data_df['SK_ID_CURR'] == cur_id].copy().reset_index(drop=True)

    # Si le client a été changé (IS_SUBMITTED existe)
    else:

        # Récupère l'id depuis st.session_state
        cur_id = st.session_state[SELECT_CLIENT]

        # Crée current client en l'initialisant depuis test_data avec le client sélectionné
        current_client_df = test_data_df[test_data_df['SK_ID_CURR'] == cur_id].copy().reset_index(drop=True)

        # Récupère les informations des widgets numeriques et checkbox pour renseigner current_client_df
        for col in current_client_df.columns:
            if (col not in ONEHOTENCODED_LIST) and (col != 'TARGET') and (col != 'SK_ID_CURR'):
                current_client_df[col] = st.session_state[col]

        # Initialise toutes les colonnes onehotencoder à False
        for col in ONEHOTENCODED_LIST:
            current_client_df[col] = False

        # Récupère les informations depuis les selectbox
        for col in ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS", "WALLSMATERIAL_MODE", "NAME_TYPE_SUITE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "WEEKDAY_APPR_PROCESS_START"]:
            valeur = st.session_state[col]
            newcol = col+"_"+valeur
            if newcol in current_client_df.columns:
                current_client_df[newcol] = True

    return current_client_df


# -------------------------------------------------------------------------
# Définitions des callback sur widgets
# -------------------------------------------------------------------------

def change_client():
    '''Callback appelée sur sélection d'un autre client dans la liste'''

    # print("change_client: reset st.session_state")

    # Reset tous les éléments dans Session state à l'exception de l'identifiant du nouveau client
    for key in st.session_state.keys():
        if key != SELECT_CLIENT:
            del st.session_state[key]


# --------------------------------------------------------

def click_submit_button():
    '''Callback appelée sur click sur le bouton submit ou lors des entrées dans les zones d'édition'''

    # Si le flag IS_SUBMITTED n'existe pas : le crée
    if IS_SUBMITTED not in st.session_state:
        st.session_state[IS_SUBMITTED] = True


# --------------------------------------------------------
# Définitions des pages
# --------------------------------------------------------

def cartouche_client(current_client_df, test_data_df, y_proba, y_pred):
    '''Affichage du cartouche client (avec TARGET et probabilité)'''

    # print("dans cartouche_client")

    with st.container(border=True):

        incol1, incol2, incol3, incol4 = st.columns(4)
        with incol1:
            st.write("Client :")
        with incol2:
            # La combobox de sélection du client
            st.selectbox(label           ="Client :",
                         help            ="Sélectionner un client pour un crédit dans la liste",
                         options         =test_data_df['SK_ID_CURR'].sort_values(),
                         key             =SELECT_CLIENT,
                         on_change       =change_client,
                         label_visibility="collapsed")
        with incol3:
            if IS_SUBMITTED not in st.session_state:
                st.write(f"TARGET (initiale) : {y_pred}")
            else:
                st.write(f"Nouvelle TARGET : {y_pred}")
        with incol4:
            if IS_SUBMITTED not in st.session_state:
                st.write(f"Probabilité (initiale) : {y_proba:.4f} / 0.27")
            else:
                st.write(f"Nouvelle probabilité : {y_proba:.4f} / 0.27")


# --------------------------------------------------------


def formulaire(current_client_df):
    '''Affichage du formulaire de saisie/modification des données client'''

    # print("dans formulaire")

    LARGEUR_COL1 = 1
    LARGEUR_COL2 = 4

    st.write("### Formulaire de saisie des données client")

    with st.form("Formulaire de saisie des données client"):

        widget_list    = []  # identifiant des widgets des zones de saisie des données client

        for col in current_client_df.columns.sort_values():
            if col != 'TARGET':

                if col in ONEHOTENCODED_LIST:

                    the_list, start_col = get_var_from_col(col)

                    # On ne crée le widget que pour la première valeur de la liste
                    if start_col not in widget_list:

                        widget_list.append(start_col)

                        # Passe en revue toutes les valeurs possible de la liste
                        optionslist = []
                        selected    = "Autre"
                        for opt in the_list:
                            if current_client_df[opt][0]:                   # Vérifie si l'option est celle sélectionnée
                                selected = opt[len(start_col)+1:]
                            optionslist.append(opt[len(start_col)+1:])      # Ajoute à la nouvelle liste le noms sans le début
                        optionslist.append("Autre")                         # Ajoute systématiquement le cas "Autre"

                        # Place la valeur sélectionnée dans st.session_state avant de définir le widget pour éviter un rechargement de page intempestif
                        if start_col not in st.session_state:
                            st.session_state[start_col] = selected

                        incol1, incol2 = st.columns([LARGEUR_COL1, LARGEUR_COL2])
                        with incol1:
                            st.write(f"{start_col} :")

                        with incol2:
                            st.selectbox(label=start_col+" :", key=start_col, label_visibility="collapsed", options=optionslist, help="Sélectionner une valeur dans la liste")

                elif (str(current_client_df[col].dtypes) == 'bool') or ((col[:5] == 'FLAG_') and (str(current_client_df[col].dtypes) == 'int64')):

                    incol1, incol2 = st.columns([LARGEUR_COL1, LARGEUR_COL2])
                    with incol2:
                        st.checkbox(col, key=col, value=current_client_df[col][0])

                elif str(current_client_df[col].dtypes) in ['float64', 'int64']:

                    incol1, incol2 = st.columns([LARGEUR_COL1, LARGEUR_COL2])
                    with incol1:
                        st.write(f"{col} :")

                    with incol2:
                        st.number_input(col, value=current_client_df[col][0], key=col, label_visibility="collapsed")
                        widget_list.append(col)

                else:
                    incol1, incol2 = st.columns([LARGEUR_COL1, LARGEUR_COL2])
                    with incol2:
                        st.write(f"{col} ({current_client_df[col].dtypes})= {current_client_df[col][0]}")

        st.form_submit_button("Faire une nouvelle prédiction", on_click=click_submit_button)


# --------------------------------------------------------

def graphiques_client(current_client_df, prepared_data_df):
    '''Affiche des graphiques et des KPI divers avec positionnement du client en cours'''

    # print("dans graphiques_client")

    # Initialise les variables pour le graphique de corrélation
    if VARIABLE_X not in st.session_state:
        st.session_state[VARIABLE_X] = CORRELATION_POSSIBLE_VAR[0]      # Par défaut : prend la 1ere variable de la liste
    if VARIABLE_Y not in st.session_state:
        st.session_state[VARIABLE_Y] = CORRELATION_POSSIBLE_VAR[-1]     # Par défaut : prend la dernière variable de la liste

    # Calculs et application des filtres pour affichage des visualisations
    distributions_df, boolean_df, kpi_df = data_transform(prepared_data_df)

    with st.container(border=False):
        # Récupère les valeurs du client sélectionné (pour affichage du positionnement du client sur les graphiques)
        age            = 0
        anciennete     = 0
        mt_credit      = 0
        mt_goods       = 0
        ext_sources    = 0
        # own_car        = ""
        education_type = ""
        income_type    = ""
        family_status  = ""
        wallsmaterial  = ""
        # valeurs quantitatives
        age            = abs(current_client_df['DAYS_BIRTH_y'][0])/365
        anciennete     = abs(current_client_df['DAYS_EMPLOYED'][0])/365
        mt_credit      = current_client_df['AMT_CREDIT'][0]
        mt_goods       = current_client_df['AMT_GOODS_PRICE'][0]
        ext_sources    = current_client_df['EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3'][0]
        obs_30_cnt     = current_client_df['OBS_30_CNT_SOCIAL_CIRCLE'][0]
        own_car_age    = current_client_df['OWN_CAR_AGE'][0]
        # valeurs qualitatives
        # own_car        = current_client_df['FLAG_OWN_CAR'][0]
        education_type = get_onehotencoded_value(current_client_df, NAME_EDUCATION_TYPE_LIST)
        income_type    = get_onehotencoded_value(current_client_df, NAME_INCOME_TYPE_LIST)
        family_status  = get_onehotencoded_value(current_client_df, NAME_FAMILY_STATUS_LIST)
        wallsmaterial  = get_onehotencoded_value(current_client_df, WALLSMATERIAL_MODE_LIST)

        # -----------------------------------------
        # Affichage des valeurs moyennes
        # -----------------------------------------
        st.subheader("Valeurs moyennes", anchor=False, help="Zone d'affichage des indicateurs concernant les principales valeurs quantitatives")
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                draw_kpi(prepared_data_df['AMT_CREDIT'][0],                 mt_credit, "AMT_CREDIT")
            with col2:
                draw_kpi(prepared_data_df['AMT_GOODS_PRICE'][0],            mt_goods, "AMT_GOODS_PRICE")
            with col3:
                draw_kpi(prepared_data_df['OBS_30_CNT_SOCIAL_CIRCLE'][0],   obs_30_cnt, "OBS_30_CNT_SOCIAL_CIRCLE")
            with col4:
                draw_kpi(prepared_data_df['OWN_CAR_AGE'][0],                own_car_age, "OWN_CAR_AGE")

        col1, col2 = st.columns([3, 1])

        with col1:
            # -----------------------------------------
            # Affichage des graphiques
            # -----------------------------------------
            st.subheader("Valeurs quantitatives", anchor=False, help="Zone d'affichage des graphiques concernant les valeurs quantitatives")
            with st.container(border=True):

                with st.container(border=False):
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        with st.container(border=True):
                            min_value = prepared_data_df['AMT_CREDIT'].min()
                            max_value = prepared_data_df['AMT_CREDIT'].max()
                            st.slider("Fourchette de valeurs pour AMT_CREDIT", key=SLIDER_AMT_CREDIT, min_value=min_value, max_value=max_value, value=[min_value, max_value])
                            graph_distrib(distributions_df, 'AMT_CREDIT', mt_credit, 'Montant (en $)', "Effet du montant du crédit sur le remboursement", True)
                    with subcol2:
                        with st.container(border=True):
                            min_value = prepared_data_df['age'].min()
                            max_value = prepared_data_df['age'].max()
                            st.slider("Fourchette de valeurs pour age", key=SLIDER_AGE, min_value=min_value, max_value=max_value, value=[min_value, max_value])
                            graph_distrib(distributions_df, 'age', age, 'Age (années)', "Effet de l'âge sur le remboursement", False)

                with st.container(border=False):
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        with st.container(border=True):
                            min_value = prepared_data_df['AMT_CREDIT'].min()
                            max_value = prepared_data_df['AMT_CREDIT'].max()
                            st.slider("Fourchette de valeurs pour AMT_GOODS_PRICE", key=SLIDER_AMT_GOODS_PRICE, min_value=min_value, max_value=max_value, value=[min_value, max_value])
                            graph_distrib(distributions_df, 'AMT_GOODS_PRICE', mt_goods,    'Montant (en $)',      "Effet du prix du bien sur le remboursement",                 True)
                    with subcol2:
                        with st.container(border=True):
                            min_value = prepared_data_df['anciennete'].min()
                            max_value = prepared_data_df['anciennete'].max()
                            st.slider("Fourchette de valeurs pour anciennete", key=SLIDER_ANCIENNETE, min_value=min_value, max_value=max_value, value=[min_value, max_value])
                            graph_distrib(distributions_df, 'anciennete', anciennete, 'Anciennete (années)', "Effet de l'anciennete sur le remboursement", False)

                with st.container(border=False):
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        with st.container(border=True):
                            min_value = prepared_data_df['EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3'].min()
                            max_value = prepared_data_df['EXT_SOURCE_1EXT_SOURCE_2EXT_SOURCE_3'].max()
                            st.slider("Fourchette de valeurs pour EXT_SOURCE_1 à 3", key=SLIDER_EXT_SOURCES, min_value=min_value, max_value=max_value, value=[min_value, max_value])
                            graph_distrib(distributions_df, 'EXT_SOURCES', ext_sources, 'Montant (en $)', "Effet de la variable EXT_SOURCE_1 à 3 sur le remboursement", False)
                    with subcol2:
                        with st.container(border=True):
                            st.selectbox(label="Variable en X", key=VARIABLE_X, label_visibility="visible", options=CORRELATION_POSSIBLE_VAR, help="Sélectionner une variable dans la liste")
                            st.selectbox(label="Variable en Y", key=VARIABLE_Y, label_visibility="visible", options=CORRELATION_POSSIBLE_VAR, help="Sélectionner une variable dans la liste")
                            graph_bivariee(prepared_data_df)

        with col2:
            # -----------------------------------------
            # Affichage des cammenberts
            # -----------------------------------------
            st.subheader("Valeurs qualitatives", anchor=False, help="Zone d'affichage des graphiques concernant les valeurs qualitatives")
            with st.container(border=True):
                # graph_boolean(boolean_df, "FLAG",                ['FLAG_OWN_CAR'],          own_car,        "Possession d'une voiture")
                graph_boolean(boolean_df, "NAME_EDUCATION_TYPE", NAME_EDUCATION_TYPE_LIST,  education_type, "Niveau scolaire des clients")
                graph_boolean(boolean_df, "NAME_INCOME_TYPE",    NAME_INCOME_TYPE_LIST,     income_type,    "Origine des revenus des clients")
                graph_boolean(boolean_df, "NAME_FAMILY_STATUS",  NAME_FAMILY_STATUS_LIST,   family_status,  "Statut familiale")
                graph_boolean(boolean_df, "WALLSMATERIAL_MODE",  WALLSMATERIAL_MODE_LIST,   wallsmaterial,  "Lieu de vie")


# --------------------------------------------------------

def graphiques_shap(current_client_df, test_data_df, explainer, y_proba, y_pred):
    '''Affiche des graphiques et des KPI divers avec positionnement du client en cours'''

    NBR_SHAP_VAR = 20

    with st.container(border=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            # Affichage du graphique shap d'explication local
            X = current_client_df.drop(columns="TARGET")
            shap_values = explainer(X)
            st.write("### Explication pour ce crédit")
            # shap.force_plot(explainer.expected_value, shap_values.values[0], X, matplotlib=True, show=False)
            shap.plots.bar(shap_values[0], max_display=NBR_SHAP_VAR, show=False)
            with st.container(border=True):
                st.pyplot(plt.gcf())
            plt.close()
            st.write(f"Base value={explainer.expected_value}")
            st.write(f"Probabilité={y_proba} / Seuil={BEST_THRESHOLD}")

        with col2:
            # Affichage du graphique shap d'explication global
            st.write(f"### Explications globales des {NBR_SHAP_VAR} variables les plus importantes")
            with st.container(border=True):
                incol1, incol2 = st.columns(2)
                with incol1:
                    shap.summary_plot(global_shap_values, test_data_df.drop(columns="TARGET"), max_display=NBR_SHAP_VAR, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                with incol2:
                    # Affichage du tableau des valeurs shap d'explication global
                    shap_df         = pd.DataFrame(global_shap_values, columns=global_feature_names)
                    vals            = shap_df.values.mean(0)
                    abs_vals        = np.abs(shap_df.values).mean(0)
                    shap_importance = pd.DataFrame(list(zip(global_feature_names, vals, abs_vals)), columns=['Variable', 'Importance', 'ImportanceABS'])
                    shap_importance.sort_values(by=['ImportanceABS'], ascending=False, inplace=True)
                    st.table(shap_importance.set_index('Variable')[['Importance']][:NBR_SHAP_VAR])


# -------------------------------------------------------------------------------------------------------------------

@st.fragment()
def fragment(prepared_data_df, test_data_df, explainer):

    # Récupère le client en cours (depuis session_state)
    current_client_df = get_current_client(test_data_df)

    # Récupère les prédictions en appelant l'API
    y_proba, y_pred = get_prediction(current_client_df)

    cartouche_client(current_client_df, test_data_df, y_proba, y_pred)

    tab1, tab2, tab3 = st.tabs(["Graphiques", "Explications SHAP", "Formulaire de saisie"])
    with tab1:
        graphiques_client(current_client_df, prepared_data_df)
    with tab2:
        graphiques_shap(current_client_df, test_data_df, explainer, y_proba, y_pred)
    with tab3:
        formulaire(current_client_df)


# -------------------------------------------------------------------------------------------------------------------
# La fonction MAIN
# Note : ce fichier est prévu pour être exécuté avec streamlit run : il n'y a donc pas le traditionnel
#   if __name__ == "__main__":
#           main()
# -------------------------------------------------------------------------------------------------------------------

# Initialization
st.set_page_config(layout="wide")   # Utilise toue la largeur disponible

# Chargement des données depuis le fichier ZIP créés lors de la préparation des données
print("Chargement des données préparées ...")
prepared_data_df = pd.read_csv(ZIP_PREPARED_DATA_FILENAME, sep=',', encoding='utf-8', compression='zip')

# Chargement des données de test depuis le fichier ZIP créés dans le projet n°7
print("Chargement des données de test ...")
test_data_df = pd.read_csv(ZIP_TEST_DATA_FILENAME, sep=',', encoding='utf-8', compression='zip')

# Chargement de l'explainer SHAP via pickel
print("Chargement de l'explainer SHAP ...")
with zipfile.ZipFile(ZIP_SHAP_EXPLAINER, 'r', zipfile.ZIP_DEFLATED) as zipf:
    with zipf.open(INSIDE_ZIP_SHAP_EXPLAINER, 'r') as file:
        explainer = pickle.load(file)

print("Chargement des shap_values associés à l'explainer ...")
with open(SHAPVALUES_DATA_FILENAME, 'rb') as file:
    global_shap_values   = pickle.load(file)
    global_feature_names = test_data_df.drop(columns='TARGET').columns

print("Démarrage du dashboard ...")

# Zone de titre et d'entete
st.write("## Projet 8-Réalisez un dashboard et assurez une veille technique")
st.write("**Script Python réalisé par Dominique LEVRAY en Août/Septembre 2024**")

# Appel le fragment (on ne rechargera pas les données après chaque modif)
fragment(prepared_data_df, test_data_df, explainer)
