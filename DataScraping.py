import requests 
from bs4 import BeautifulSoup
import pandas as pd
response = request.get("https://books.toScrape.com")
html_content=response.text
print(html_content)

if html_content:
  soup = BeautifulSoup(html_content,'html.parser')
  print(soup)
  books = soup.find_all('articles',class='product_pod')
else:
  print("no data fetched")
data=[]
print("Details of book")
for book in books:
  title=book.find('h3').find('a')['title']
  price=book.find('p', class="price_color").text
  data.append([title,price'])

df=pd.DataFrame(data,columns=['title','price'])
print(df)
df.to_CSV('books_data.csv',index=false)
