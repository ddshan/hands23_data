import os, glob


def add_video(file_path, width=400, is_lazy=False, replace_from='', replace_to=''):


    html = f''' 
                <td>
                <video class='lazy' width={width} controls autoplay loop muted playsinline, poster=''>
                <source data-src={file_path} type='video/mp4'>
                </video>
                </td>
            '''
    return html



header = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        # table {{
        #     font-family: arial, sans-serif;
        #     # border-collapse: collapse;
        #     width: 100%;
        # }}

        # td, th {{
        #     border: 1px solid black;
        #     text-align: left;
        #     padding: 4px;
        # }}

        # tr:nth-child(even) {{
        #     background-color: #dddddd;
        # }}
        # tr:hover {{
        #     background-color: lightyellow;
        # }}
        </style>
        </head>
        <body>

'''

tailer  = f'''
    <script src="https://cdn.jsdelivr.net/npm/vanilla-lazyload@17.8.3/dist/lazyload.min.js"></script>   
    <script src='lazyload.js'></script>
    </body>
    </html>

'''    

if __name__ == '__main__':
# vis
    import glob
    
    
    
    img_list = glob.glob(f'vis/handnew_sam/*.jpg')
    content_ls = glob.glob(f'vis/videos_plot/*.mp4')
    html_path = f'vis_sam_video.html'


    # with open(html_path, 'w') as f_viz:
    #     f_viz.write(f'<h2> Segment Angything predictions </h2>')
    #     print(len(img_list))
    #     print(len(video_ls))

    #     f_viz.write("<table border='2'>")
    #     for ind,i in enumerate(range(0, len(img_list), 5)):
    #         f_viz.write("<tr>")åç
    #         f_viz.write(f'<td>{ind}</td>')
    #         for j in range(5):
    #             if i+j >= len(img_list): continue
    #             # img_path = os.path.join('./', img_list[i+j])
    #             img_path = img_list[i+j]#.replace(visroot, '.')
    #             f_viz.write(f"<td><image src={img_path} lazy width=400></td>") # raw
    #             # f_viz.write(f"<td>{img_path.split('/')[-1]}</td>") # raw
    #         f_viz.write('</tr>')
    #     f_viz.write('</table>') 


    with open(html_path, 'w') as f_viz:
        f_viz.write(header)

        f_viz.write(f'<h2> Segment Angything predictions </h2>')
        print(len(img_list))
        print(len(content_ls))

        f_viz.write("<table border='2'>")
        for ind,i in enumerate(range(0, len(content_ls), 5)):
            f_viz.write("<tr>")
            f_viz.write(f'<td>{ind}</td>')
            for j in range(5):
                if i+j >= len(content_ls): continue
                file_path = content_ls[i+j]#.replace(visroot, '.')
                # f_viz.write(f"<td><image src={file_path} lazy width=400></td>") # raw

                f_viz.write(add_video(file_path))
               
            f_viz.write('</tr>')
        f_viz.write('</table>') 
        f_viz.write(tailer)
