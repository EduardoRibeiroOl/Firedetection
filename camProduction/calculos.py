
#Àrea de captura (area da janela da camera) Captura de APROXIMADAMENTE 640x480 no meu notebook, 
# mas isso pode variar dependendo do dispositivo

def ponto_medio_box(x1, y1, x2, y2):
    #  ponto médio da caixa (centro da caixa)
    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2
    
    return centro_x, centro_y

# ângulo = (centro_x - largura/2) * (HFOV / largura)

#   0 px               320 px                 640 px
#   |--------------------|----------------------|
#   Esquerda            Centro               Direita

#HFOV = 2arctan(largura visivel/2​)/(distancia)
# de exemplo pega logo 70° de fov aí

def optica(centro_x, largura, HFOV):
    # angulo das coisas, qualquer coisa refatora
    
    centro_imagem = largura / 2
    angle = (centro_x - centro_imagem) * HFOV / largura
    
    return angle
