#!/bin/bash
# This script assumes that Homebrew is installed on your system.

REDISJSON_VERSION="2.8.3"
REDIS_JSON_MODULE="build/RedisJSON-$REDISJSON_VERSION/target/release/librejson.dylib"

if [ -f "$REDIS_JSON_MODULE" ]; then
    echo "$REDIS_JSON_MODULE already built, simply running starting the Redis server."
else 
    echo "$REDIS_JSON_MODULE not found, building RedisJSON."

    # Install Redis
    mkdir -p build
    cd build

    if [[ $(brew list | grep rust) ]]; then
        echo "Rust is already installed."
    else
        echo "Rust not found, installing Rust."
        brew install rust
    fi

    if [[ $(brew list | grep redis) ]]; then
        echo "Redis is already installed."
    else
        echo "Redis not found, installing Redis."
        brew install redis
    fi

    REDISJSON_VERSION="2.8.3"
    wget https://github.com/RedisJSON/RedisJSON/archive/refs/tags/v$REDISJSON_VERSION.zip -O RedisJSON-$REDISJSON_VERSION.zip
    unzip RedisJSON-$REDISJSON_VERSION.zip

    cd RedisJSON-$REDISJSON_VERSION
    cargo build --release

    cd ..

    touch redis.conf
    echo "bind 0.0.0.0" >> redis.conf
    echo "loadmodule $REDIS_JSON_MODULE" >> redis.conf

    cd ..
fi

# To run Redis with RedisJSON
redis-server build/redis.conf