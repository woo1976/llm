# %%capture --no-stderr
# %pip install -U chromadb

import os
import json
import requests
from urllib.parse import urlparse
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
from tiktoken import encoding_for_model

dir_path = "/content/drive/MyDrive/Colab_Notebooks/"

# API keys
openai_file = "googlecolab_openai_key.txt"
with open(dir_path + openai_file, "r") as file:
    openai_api_key = file.read()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=dir_path+"+repo_search/chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction() # default: ll-MiniLM-L6-v2

def get_repo_info(repo_url):
    """
    Extract the owner and repository name from the GitHub URL.
    """
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 2:
        raise ValueError("Invalid GitHub repository URL.")
    owner = path_parts[0]
    repo = path_parts[1].replace('.git', '')
    return owner, repo

def get_file_content(download_url):
    """
    Retrieve the raw content of a file.
    """
    response = requests.get(download_url)
    response.raise_for_status()
    return response.text

def get_contents(owner="woo1976", repo="jenkins-python-test", path="", branch="master"):
    """
    Recursively retrieve the contents of the repository, excluding image and media files.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": branch}  # You can change the branch if needed
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    items = response.json()
    contents = []

    if not isinstance(items, list):
        items = [items]

    # Exclude image and media formats
    excluded_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.mp4', '.mp3', '.wav', '.avi', '.mov']

    for item in items:
        if item['type'] == 'file' and any(item['path'].endswith(ext) for ext in excluded_extensions):
            continue

        if item['type'] == 'file':
            file_content = get_file_content(item['download_url'])
            contents.append({
                'type': 'file',
                'path': item['path'],
                'content': file_content
            })
        elif item['type'] == 'dir':
            dir_contents = get_contents(owner, repo, item['path'], branch)
            contents.append({
                'type': 'dir',
                'path': item['path'],
                'contents': dir_contents
            })
    return contents

# Test
res = get_contents

# [{'type': 'file',
#   'path': '.behaverc',
#   'content': '[behave]\r\npaths=tests/features'},
#  {'type': 'file',
#   'path': '.gitignore',
#   'content': '# Byte-compiled / optimized / DLL files\r\n__pycache__/\r\n*.py[cod]\r\n*$py.class\r\n\r\n# C extensions\r\n*.so\r\n\r\n# Distribution / packaging\r\n.Python\r\nenv/\r\nbuild/\r\ndevelop-eggs/\r\ndist/\r\ndownloads/\r\neggs/\r\n.eggs/\r\nlib/\r\nlib64/\r\nparts/\r\nsdist/\r\nvar/\r\nwheels/\r\n*.egg-info/\r\n.installed.cfg\r\n*.egg\r\n\r\n# PyInstaller\r\n#  Usually these files are written by a python script from a template\r\n#  before PyInstaller builds the exe, so as to inject date/other infos into it.\r\n*.manifest\r\n*.spec\r\n\r\n# Installer logs\r\npip-log.txt\r\npip-delete-this-directory.txt\r\n\r\n# Unit test / coverage reports\r\nhtmlcov/\r\n.tox/\r\n.coverage\r\n.coverage.*\r\n.cache\r\nnosetests.xml\r\ncoverage.xml\r\n*.cover\r\n.hypothesis/\r\n.pytest_cache/\r\ntest-reports/\r\n\r\n# Translations\r\n*.mo\r\n*.pot\r\n\r\n# Django stuff:\r\n*.log\r\nlocal_settings.py\r\n\r\n# Flask stuff:\r\ninstance/\r\n.webassets-cache\r\n\r\n# Scrapy stuff:\r\n.scrapy\r\n\r\n# Sphinx documentation\r\ndocs/_build/\r\n\r\n# PyBuilder\r\ntarget/\r\n\r\n# Jupyter Notebook\r\n.ipynb_checkpoints\r\n\r\n# pyenv\r\n.python-version\r\n\r\n# celery beat schedule file\r\ncelerybeat-schedule\r\n\r\n# SageMath parsed files\r\n*.sage.py\r\n\r\n# dotenv\r\n.env\r\n\r\n# virtualenv\r\n.venv\r\nvenv/\r\nENV/\r\n\r\n# Spyder project settings\r\n.spyderproject\r\n.spyproject\r\n\r\n# Rope project settings\r\n.ropeproject\r\n\r\n# mkdocs documentation\r\n/site\r\n\r\n# mypy\r\n.mypy_cache/\r\n'},
#  {'type': 'file',
#   'path': 'Jenkinsfile',
#   'content': 'pipeline {\r\n    agent any\r\n\r\n    triggers {\r\n        pollSCM(\'*/5 * * * 1-5\')\r\n    }\r\n\r\n    options {\r\n        skipDefaultCheckout(true)\r\n        // Keep the 10 most recent builds\r\n        buildDiscarder(logRotator(numToKeepStr: \'10\'))\r\n        timestamps()\r\n    }\r\n\r\n    environment {\r\n      PATH=\'%PATH%;C:\\\\Program Files (x86)\\\\Jenkins\\\\miniconda3;C:\\\\Program Files (x86)\\\\Jenkins\\\\miniconda3\\\\Scripts\'\r\n    }\r\n\r\n    stages {\r\n        stage(\'Check path\') {\r\n            steps{\r\n                echo %PATH%\r\n            }    \r\n        }\t\t\t\t\r\n        stage ("Code pull"){\r\n            steps{\r\n                checkout scm\r\n            }\r\n        }\r\n\r\n        stage(\'Build environment\') {\r\n            steps {\r\n                echo "Building virtualenv"\r\n                sh  \'\'\' conda create --yes -n ${BUILD_TAG} python\r\n                        source activate ${BUILD_TAG}\r\n                        pip install -r requirements/dev.txt\r\n                    \'\'\'\r\n            }\r\n        }\r\n\r\n        stage(\'Static code metrics\') {\r\n            steps {\r\n                echo "Raw metrics"\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        radon raw --json irisvmpy > raw_report.json\r\n                        radon cc --json irisvmpy > cc_report.json\r\n                        radon mi --json irisvmpy > mi_report.json\r\n                        sloccount --duplicates --wide irisvmpy > sloccount.sc\r\n                    \'\'\'\r\n                echo "Test coverage"\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        coverage run irisvmpy/iris.py 1 1 2 3\r\n                        python -m coverage xml -o reports/coverage.xml\r\n                    \'\'\'\r\n                echo "Style check"\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        pylint irisvmpy || true\r\n                    \'\'\'\r\n            }\r\n            post{\r\n                always{\r\n                    step([$class: \'CoberturaPublisher\',\r\n                                   autoUpdateHealth: false,\r\n                                   autoUpdateStability: false,\r\n                                   coberturaReportFile: \'reports/coverage.xml\',\r\n                                   failNoReports: false,\r\n                                   failUnhealthy: false,\r\n                                   failUnstable: false,\r\n                                   maxNumberOfBuilds: 10,\r\n                                   onlyStable: false,\r\n                                   sourceEncoding: \'ASCII\',\r\n                                   zoomCoverageChart: false])\r\n                }\r\n            }\r\n        }\r\n\r\n\r\n\r\n        stage(\'Unit tests\') {\r\n            steps {\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        python -m pytest --verbose --junit-xml reports/unit_tests.xml\r\n                    \'\'\'\r\n            }\r\n            post {\r\n                always {\r\n                    // Archive unit tests for the future\r\n                    junit allowEmptyResults: true, testResults: \'reports/unit_tests.xml\'\r\n                }\r\n            }\r\n        }\r\n\r\n        stage(\'Acceptance tests\') {\r\n            steps {\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        behave -f=formatters.cucumber_json:PrettyCucumberJSONFormatter -o ./reports/acceptance.json || true\r\n                    \'\'\'\r\n            }\r\n            post {\r\n                always {\r\n                    cucumber (buildStatus: \'SUCCESS\',\r\n                    fileIncludePattern: \'**/*.json\',\r\n                    jsonReportDirectory: \'./reports/\',\r\n                    parallelTesting: true,\r\n                    sortingMethod: \'ALPHABETICAL\')\r\n                }\r\n            }\r\n        }\r\n\r\n        stage(\'Build package\') {\r\n            when {\r\n                expression {\r\n                    currentBuild.result == null || currentBuild.result == \'SUCCESS\'\r\n                }\r\n            }\r\n            steps {\r\n                sh  \'\'\' source activate ${BUILD_TAG}\r\n                        python setup.py bdist_wheel\r\n                    \'\'\'\r\n            }\r\n            post {\r\n                always {\r\n                    // Archive unit tests for the future\r\n                    archiveArtifacts allowEmptyArchive: true, artifacts: \'dist/*whl\', fingerprint: true\r\n                }\r\n            }\r\n        }\r\n\r\n        // stage("Deploy to PyPI") {\r\n        //     steps {\r\n        //         sh """twine upload dist/*\r\n        //         """\r\n        //     }\r\n        // }\r\n    }\r\n\r\n    post {\r\n        always {\r\n            sh \'conda remove --yes -n ${BUILD_TAG} --all\'\r\n        }\r\n        failure {\r\n            emailext (\r\n                subject: "FAILED: Job \'${env.JOB_NAME} [${env.BUILD_NUMBER}]\'",\r\n                body: """<p>FAILED: Job \'${env.JOB_NAME} [${env.BUILD_NUMBER}]\':</p>\r\n                         <p>Check console output at &QUOT;<a href=\'${env.BUILD_URL}\'>${env.JOB_NAME} [${env.BUILD_NUMBER}]</a>&QUOT;</p>""",\r\n                recipientProviders: [[$class: \'DevelopersRecipientProvider\']])\r\n        }\r\n    }\r\n}\r\n'},

# Continue.... TODO