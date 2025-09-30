import requests

url = "https://open-weather13.p.rapidapi.com/city"

querystring = {"city":"beijing","lang":"EN"}

headers = {
	"x-rapidapi-key": "5bad9b417amsh4fb61074f2d054bp114882jsn475094fc19bb",
	"x-rapidapi-host": "open-weather13.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())