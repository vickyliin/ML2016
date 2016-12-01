import plotly.plotly as py
def plot(T):
    data = []
    for i in range(len(T['tag'].unique())):
        name = T['tag'].unique()[i]
        x = T[ T['tag'] == name ]['x']
        y = T[ T['tag'] == name ]['y']
        
        trace = dict(
            name = name,
            x = x, y = y,
            type = "scatter2d",    
            mode = 'markers',
            marker = dict( size=3, line=dict(width=0) ) )
        data.append( trace )

    title = '100D Autoencoder / %dCluster' % len(T['tag'].unique())

    layout = dict(
        width=600, height=600,
        autosize=False,
        title=title,
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            aspectratio = dict( x=1, y=1),
            aspectmode = 'manual'
        ),
    )

    fig = dict(data=data, layout=layout)
    py.image.save_as(fig, filename=''\
            .join(title.split(' '))\
            .replace('/','_')+'.png')
