
from apps import create_app
import cherrypy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = create_app()

#使用cherrypy增加并发
cherrypy.tree.graft(app.wsgi_app,'/')
cherrypy.config.update({
    "server.socket_host":"0.0.0.0",
    "server.socket_port":8081,
    "engine.autoreload.on":True
})


