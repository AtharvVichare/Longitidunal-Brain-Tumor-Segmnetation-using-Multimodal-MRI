pipeline {
    agent any

    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
        ansiColor('xterm')
    }

    environment {
        PYTHONUNBUFFERED = '1'
        VENV_DIR = '.venv'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Environment Setup') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            python3 -m venv ${VENV_DIR}
                            . ${VENV_DIR}/bin/activate
                            pip install --upgrade pip
                            pip install flake8 pytest pytest-cov
                            pip install -r requirements.txt
                        '''
                    } else {
                        bat '''
                            python -m venv %VENV_DIR%
                            call %VENV_DIR%\\Scripts\\activate.bat
                            python -m pip install --upgrade pip
                            pip install flake8 pytest pytest-cov
                            pip install -r requirements.txt
                        '''
                    }
                }
            }
        }

        stage('Lint & Code Quality') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            . ${VENV_DIR}/bin/activate
                            flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
                            flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
                        '''
                    } else {
                        bat '''
                            call %VENV_DIR%\\Scripts\\activate.bat
                            flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
                            flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
                        '''
                    }
                }
            }
        }

        stage('Syntax & Unit Tests') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            . ${VENV_DIR}/bin/activate
                            python -m compileall -q .
                            if [ -d "tests" ]; then
                                pytest tests/ --junitxml=reports/junit.xml --cov=. --cov-report=xml:reports/coverage.xml
                            else
                                echo "No tests folder found. Running smoke import tests..."
                                python -c "import torch; import monai; import train; import evaluate; print('Smoke test passed!')"
                            fi
                        '''
                    } else {
                        bat '''
                            call %VENV_DIR%\\Scripts\\activate.bat
                            python -m compileall -q .
                            if exist tests (
                                pytest tests/ --junitxml=reports/junit.xml --cov=. --cov-report=xml:reports/coverage.xml
                            ) else (
                                echo No tests folder found. Running smoke import tests...
                                python -c "import torch; import monai; import train; import evaluate; print('Smoke test passed!')"
                            )
                        '''
                    }
                }
            }
        }

        stage('Dry Run / Sanity Check') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            . ${VENV_DIR}/bin/activate
                            python train.py --help || true
                            python evaluate.py --help || true
                        '''
                    } else {
                        bat '''
                            call %VENV_DIR%\\Scripts\\activate.bat
                            python train.py --help || exit /b 0
                            python evaluate.py --help || exit /b 0
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            cleanWs(deleteDirs: true, notFailBuild: true)
        }
        success {
            echo "Pipeline completed successfully!"
        }
        failure {
            echo "Pipeline failed. Check build logs for details."
        }
    }
}
