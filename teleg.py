import telebot
import schedule
import time
from models import *
from news_script import *
from dotenv import load_dotenv
import telebot
import os
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
bot = telebot.TeleBot(API_TOKEN)

# All useer preferences. Currently they're stored as global variable, but better approach to store in db
user_preferences = {}


# Handle the /start command
@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "Hi! Use /setgenres to set your preferred news genres.")


@bot.message_handler(commands=["stop"])
def stop_updates(message):
    user_id = message.from_user.id
    if user_id in user_preferences:
        del user_preferences[user_id]
        bot.reply_to(message, "You have successfully unsubscribed from news updates.")
    else:
        bot.reply_to(message, "You are not currently subscribed to news updates.")


# Handle the /setgenres command
@bot.message_handler(commands=["setgenres"])
def set_genres(message):
    bot.reply_to(
        message, "Please provide the genres you are interested in, separated by commas."
    )
    bot.register_next_step_handler(message, process_genres)


def process_genres(message):
    user_id = message.from_user.id
    genres = " ".join(message.text.split()[:])
    user_preferences[user_id] = genres.split(",")
    bot.reply_to(message, f"Your preferred genres are set to: {genres}")



def send_news():
    for user_id, genres in user_preferences.items():
        genres = check_input(genres)
        news = get_news(genres)
        for articles in news:
            for article in articles:
                content = article["content"]
                if content is None:
                    content = article["description"]
                genre = classify(content)
                summary = summarize(content)
                url = article["url"]
                bot.send_message(
                    chat_id=user_id,
                    text=f'{summary}\n genre: {genre if genre is not None else ""}\n {url}',
                )

        bot.send_message(chat_id=user_id, text="Daily news update! 📰")



scheduler.add_job(send_news, "cron", hour=1, minute=20)
scheduler.start()

bot.polling(none_stop=True)
