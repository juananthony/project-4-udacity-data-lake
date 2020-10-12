import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek, monotonically_increasing_id
from pyspark.sql.types import TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Creates a Spark Session. If there is a previous one created, this method will return the previous SparkSession.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    This method process song data to extract the fields to create songs and artists tables.
    
    Parameters:
        - spark       : SparkSession
        - input_data  : path where the input files are persisted
        - output_data : path to persist the output tables (songs and artists)
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/*/*/*/*.json")
    
    # read song data file
    df = spark.read.json(song_data).dropDuplicates().cache()
    df.createOrReplaceTempView("song_df")

    # extract columns to create songs table
    songs_table = df.select("song_id", "title", "artist_id", "year", "duration")
    
    # write songs table to parquet files partitioned by year and artist
    songs_table = df.write.partitionBy("year", "artist_id").parquet(os.path.join(output_data, "songs"), "overwrite")

    # extract columns to create artists table
    artists_table = df.selectExpr(
        "artist_id", 
        "artist_name as name", 
        "artist_location as location",
        "artist_latitude as latitude",
        "artist_longitude as longitude")
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, "artists"), "overwrite")


def process_log_data(spark, input_data, output_data):
    """
    This method process log and data to extract fields to create time, users and songplays tables.
    
    Parameters:
        - spark       : SparkSession
        - input_data  : path where the input files are persisted
        - output_data : path to persist the output tables (songs and artists)
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, "log_data/*.json")

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(col("page") == "NextSong").cache()

    # extract columns for users table    
    users_table = df.selectExpr(
        "userId as user_id",
        "firstName as first_name",
        "lastName as last_name",
        "gender",
        "level")
    
    # write users table to parquet files
    users_table = users_table.write.parquet(os.path.join(output_data, "users"), "overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    #get_datetime = udf(lambda x: datetime.fromtimestamp(int(x) / 1000.0))
    df = df.withColumn("start_time", get_timestamp(df.ts))
    
    # extract columns to create time table
    time_table = df.select(
        col("start_time"),
        hour("start_time").alias("hour"),
        dayofmonth("start_time").alias("day"),
        weekofyear("start_time").alias("week"),
        month("start_time").alias("month"),
        year("start_time").alias("year"),
        dayofweek("start_time").alias("weekday")
    )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(os.path.join(output_data, "time"), "overwrite")

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT * FROM song_df")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df \
        .join(song_df, df.artist == song_df.artist_name, "inner") \
        .select("start_time",
                col("userId").alias("user_id"),
                "level",
                "song_id",
                "artist_id",
                col("sessionId").alias("session_id"),
                col("artist_location").alias("location"),
                col("userAgent").alias("user_agent")) \
        .withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(os.path.join(output_data, "songplays"), "overwrite")


def main():
    """
    Main method. First, it creates a spark session. Then, it processes song and log data.
    """
    spark = create_spark_session()
    input_data = config["AWS"]["INPUT_DATA"]
    output_data = config["AWS"]["OUTPUT_DATA"]
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
