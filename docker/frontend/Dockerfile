# Образ frontend: статика (HTML, CSS, JS) раздаётся через nginx
FROM nginx:alpine

# Удаляем дефолтный конфиг и подключаем свой
RUN rm -f /etc/nginx/conf.d/default.conf
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Копируем статику: index.html в корень, остальное в /static/ (как в API)
COPY app/frontend/index.html /usr/share/nginx/html/
COPY app/frontend/style.css app/frontend/chat.js /usr/share/nginx/html/static/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
