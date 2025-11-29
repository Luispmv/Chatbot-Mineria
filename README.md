
# Chatbot con datos de telecomunicaciones

Clonamos el proyecto con el siguiente comando

```python
git clone https://github.com/Luispmv/Chatbot-Mineria.git
```

Una vez tengas clonado el repositorio entra en la carpeta
```python
cd Chatbot-Mineria
```

Dentro de la carpeta crea un entorno virtual con el siguiente comando

```python
python3 -m venv env
```

Activa tu entorno virtual
```python
Linux 💪
source env/bin/activate 

Powershell
.\env\Scripts\Activate.ps1
```

Instala las dependencias del proyecto
```python
pip install -r requirements.txt
```

Dirigete a esta pagina para obtener un API KEY de Gemini
https://aistudio.google.com/api-keys


Una vez tengas tu API key crea un archivo .env con lo siguiente
```python
GEMINI_API_KEY=tuapikey
```

Por ultimo ejecuta el proyecto con el siguiente comando.
```python
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
