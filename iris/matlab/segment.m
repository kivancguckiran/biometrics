function segment()
    segmentFiles('../iris/', 'segmentedIris/');
    segmentFiles('../casia/', 'segmentedCasia/');
end

function segmentFiles(path, segmentPath)
    listing = dir(path);
    listing = listing(3:end);
    for i = 1:size(listing)
        file = listing(i).name;
        eye = imread(strcat(path, file));
        [~, ~, out] = thresh(eye, 50, 400);
        fig = figure;
        set(fig, 'Visible', 'off');
        imshow(out)
        saveas(fig, strcat(segmentPath, file));
    end
end
