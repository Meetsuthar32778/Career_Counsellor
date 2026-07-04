FROM python 

WORKDIR /app

COPY . .
ADD requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "5000"]
