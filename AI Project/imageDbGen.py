import gl
import os
gl.resetdefaults()
gl.scriptformvisible(0)
gl.toolformvisible(0)
gl.bmpzoom(1)
gl.view(1)
gl.linewidth(0)
gl.colorbarposition(0)
path = b'G:\CS534_Project_DB'
out = 'G:\CS534_Project_DB\Out_PNG'
imageDirs = {
    'ALI_M_MUSLIM': True,
    'LESION_CHALLENGE': True,
    'LESJAK_SPICLIN_3D': True,
    'LESJAK_SPICLIN_LON': True,
    'OASIS_3': False,
}
for d in os.listdir(str(path.decode())):
    if d in imageDirs.keys():
        if imageDirs.get(d):
            for i in os.listdir(str(path.decode())+'\\'+str(d)+'\\NII_Images'):
                for f in os.listdir(str(path.decode())+'\\'+str(d)+'\\NII_Images\\'+str(i)):
                    gl.loadimage(str(path.decode())+'\\'+str(d)+'\\NII_Images\\'+str(i)+'\\'+str(f))
                    gl.savebmp(str(out)+'\\image_'+str(f)+'.png')
gl.quit()