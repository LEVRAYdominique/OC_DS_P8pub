# OC_DS_P8pub
Partie public du projet n°8 d'OpenClassrooms - Cursus Data Scientist 
Réalisé par Dominique LEVRAY en Août/Septembre 2024

**Réalisez un dashboard et assurez une veille technique**

Missions :
- Mission 1 :
    Cette mission est la suite directe du projet n°7 et vise à transformer le script streamlit de test de l'API en un dashboard complet
    Spécifications pour le dashboard interactif :
    - Visualisation et interprétation du score de crédit des clients :
        - Permettre de visualiser le score et sa probabilité pour chaque client de façon intelligible pour une personne non experte en data science.
        - Permettre de visualiser l'interprétation du score pour chaque client.
        - Visualisation du score de crédit (jauge colorée).
        - Visualisation de la contribution des features qui ont amené le modèle à prédire le score (feature importance locale et comparaison avec la feature importance globale).
    - Informations descriptives des clients :
        - Permettre de visualiser les principales informations descriptives relatives à un client.
        - Visualisation des caractéristiques clients et comment ils se situent par rapport aux autres clients sous forme de graphiques (possibilité de choisir feature par feature).
    - Comparaison et analyse des clients :
        - Permettre de comparer, à l'aide de graphiques, les principales informations descriptives relatives à un client à l'ensemble des clients ou à un groupe de clients similaires (via un système de filtre).
        - Graphique d'analyse bi-variée entre deux features sélectionnées.
        - Autres graphiques pertinents pour la présentation des données client.
    - Accessibilité et modifications des informations clients :
        - Prendre en compte le besoin des personnes en situation de handicap dans la réalisation des graphiques, en couvrant des critères d'accessibilité du WCAG.
        - Permettre d'obtenir un score et une probabilité actualisés après avoir saisi une modification d'une ou plusieurs informations relatives à un client.
        - Permettre de saisir un nouveau dossier client pour obtenir le score et la probabilité.
        - Interface pour la modification des informations client via l'API.
    - Intégrations et déploiement :
        - Déployer le dashboard sur une plateforme Cloud pour qu'il soit accessible à d'autres utilisateurs sur leur poste de travail.
        - Intégration d'une API pour récupérer le score du client.

**Ce dépôt contient la partie publique du projet et comprend :**
- Le dashboard sous forme d'un script python à lancer avec streamlit
- Des fichiers de contrôles pour une publication Azure Web Apps en mode "docker"
- Des fichiers de données