cat 'reserve.txt'  | cut -d ' ' -f 1 | xargs -i cp {} './train'
