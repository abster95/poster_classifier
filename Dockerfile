FROM pytorch/pytorch
RUN pip install flask imageio pandas
RUN git clone https://github.com/abster95/poster_classifier
RUN pip install -e poster_classifier
RUN mkdir poster_classifier/posters/ckpt
COPY ./posters/ckpt/best.ckpt poster_classifier/posters/ckpt
RUN python poster_classifier/posters/service/server.py
EXPOSE 8080
