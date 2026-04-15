import sys
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

# --- КОНФИГУРАЦИЯ ---
EVENTS_PATH = "/user/master/data/geo/events"
GEO_PATH = "/user/s26586465/data/geo/geo.csv"
OUTPUT_PATH = "/user/s26586465/analytics/friend_recommendations"

def main():
    spark = SparkSession.builder \
        .appName("FriendRecommendations_Final_Fixed") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    # 1. Загружаем справочник городов
    # Добавляем хак: мапим города на таймзоны, раз их нет в csv
    geo = spark.read.csv(GEO_PATH, sep=';', header=True) \
        .withColumn("city_lat", F.radians(F.regexp_replace(F.col("lat"), ",", ".").cast("double"))) \
        .withColumn("city_lon", F.radians(F.regexp_replace(F.col("lng"), ",", ".").cast("double"))) \
        .withColumn("timezone", 
            F.when(F.col("city") == "Sydney", "Australia/Sydney")
             .when(F.col("city") == "Melbourne", "Australia/Melbourne")
             .when(F.col("city") == "Brisbane", "Australia/Brisbane")
             .when(F.col("city") == "Perth", "Australia/Perth")
             .when(F.col("city") == "Adelaide", "Australia/Adelaide")
             .otherwise("Australia/Sydney") # Дефолт
        ) \
        .select(F.col("id").alias("zone_id"), "timezone", "city_lat", "city_lon")

    # 2. Список подписок
    subscriptions = spark.read.parquet(EVENTS_PATH) \
        .filter("event.subscription_channel IS NOT NULL") \
        .select(
            F.col("event.subscription_user").alias("user_id"),
            F.col("event.subscription_channel").alias("channel_id")
        ).distinct()

    # 3. Последние координаты и время пользователей
    user_last_loc = spark.read.parquet(EVENTS_PATH) \
        .filter("event.message_from IS NOT NULL AND lat IS NOT NULL") \
        .withColumn("rn", F.row_number().over(Window.partitionBy("event.message_from").orderBy(F.col("date").desc(), F.col("event.message_ts").desc()))) \
        .filter("rn = 1") \
        .select(
            F.col("event.message_from").alias("user_id"),
            F.radians(F.col("lat")).alias("lat"),
            F.radians(F.col("lon")).alias("lon"),
            F.col("event.message_ts").alias("ts")
        )

    # 4. Исключаем тех, кто уже переписывался
    direct_messages = spark.read.parquet(EVENTS_PATH) \
        .filter("event.message_from IS NOT NULL AND event.message_to IS NOT NULL") \
        .select(
            F.least(F.col("event.message_from"), F.col("event.message_to")).alias("u1"),
            F.greatest(F.col("event.message_from"), F.col("event.message_to")).alias("u2")
        ).distinct()

    # 5. Формируем пары
    pairs = subscriptions.alias("s1") \
        .join(subscriptions.alias("s2"), "channel_id") \
        .filter("s1.user_id < s2.user_id") \
        .select(F.col("s1.user_id").alias("user_left"), F.col("s2.user_id").alias("user_right")) \
        .distinct()

    pairs_filtered = pairs.join(
        direct_messages, 
        (pairs.user_left == direct_messages.u1) & (pairs.user_right == direct_messages.u2), 
        "left_anti"
    )

    # 6. Считаем расстояние между пользователями
    pairs_with_coords = pairs_filtered \
        .join(user_last_loc.alias("loc_l"), pairs_filtered.user_left == F.col("loc_l.user_id")) \
        .join(user_last_loc.alias("loc_r"), pairs_filtered.user_right == F.col("loc_r.user_id"))

    dist_pairs_expr = 2 * 6371 * F.asin(F.sqrt(
        F.pow(F.sin((F.col("loc_l.lat") - F.col("loc_r.lat")) / 2), 2) +
        F.cos(F.col("loc_l.lat")) * F.cos(F.col("loc_r.lat")) * F.pow(F.sin((F.col("loc_l.lon") - F.col("loc_r.lon")) / 2), 2)
    ))

    candidates = pairs_with_coords \
        .withColumn("distance", dist_pairs_expr) \
        .filter("distance <= 1.0") \
        .select("user_left", "user_right", "loc_l.lat", "loc_l.lon", "loc_l.ts")

    # 7. Привязка к городу для zone_id и local_time
    final_recs = candidates.crossJoin(F.broadcast(geo)) \
        .withColumn("dist_to_city", 2 * 6371 * F.asin(F.sqrt(
            F.pow(F.sin((F.col("lat") - F.col("city_lat")) / 2), 2) +
            F.cos(F.col("lat")) * F.cos(F.col("city_lat")) * F.pow(F.sin((F.col("lon") - F.col("city_lon")) / 2), 2)
        ))) \
        .withColumn("rn_city", F.row_number().over(Window.partitionBy("user_left", "user_right").orderBy("dist_to_city"))) \
        .filter("rn_city = 1") \
        .withColumn("processed_dttm", F.current_timestamp()) \
        .withColumn("local_time", F.from_utc_timestamp(F.col("ts"), F.col("timezone"))) \
        .select("user_left", "user_right", "processed_dttm", "zone_id", "local_time")

    # 8. Сохранение
    final_recs.write.mode("overwrite").parquet(OUTPUT_PATH)
    print(f"--- SUCCESS: Рекомендации сохранены в {OUTPUT_PATH} ---")

if __name__ == "__main__":
    main()