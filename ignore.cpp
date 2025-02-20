    // unsigned numTiffSlices = pTiffSlices.size();
    // assert(numTiffSlices == 33);
    // const int expandFactor = 3; 
    // unsigned numSynthSlices = expandFactor * (numTiffSlices-1) + 1;
    // std::vector<cv::Mat> iTiffSlices;

    // for (int synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
    //     int tiffSlice = int(synthSlice / expandFactor); // "real" slice index 
    //     if(synthSlice % expandFactor == 0) 
    //     { // copy the real slice to the synth one, verbatim
    //     iTiffSlices.push_back(pTiffSlices[tiffSlice]);
    //     } 
    //     else if (synthSlice % expandFactor == 1) {
    //         // Interpolate between realTiff[tiffSlice] and realTiff[tiffSlice + 1]
    //         interpolateSlices(pTiffSlices[tiffSlice], 
    //                         pTiffSlices[tiffSlice + 1], 
    //                         iTiffSlices, 
    //                         expandFactor - 1);
    //     }
    // }