/* All numbers are copied from the paper (tables, figures, and appendix). */
window.AVTR_DATA = {
  tables: {
    image: {
      groups: [
        { label: 'HighRes Image', span: 4 },
        { label: 'Visual Spatial', span: 4 },
      ],
      cols: ['V*', 'HR-Bench-4K', 'HR-Bench-8K', 'BLINK', 'ERQA', 'MMSI-Bench', 'VSI-Bench', 'RealWorldQA'],
      sections: [
        { label: 'Proprietary', reference: true, rows: [
          ['GPT-5', 72.8, null, null, 71.0, 65.7, 41.9, 55.0, 76.1],
          ['Gemini 2.5 Pro', 83.8, 87.3, 85.4, 73.5, 55.7, 38.0, 53.6, 68.7],
          ['GPT-5 mini', 78.6, 78.6, 74.4, 56.7, 54.0, 34.1, 48.7, null],
        ] },
        { label: 'Open foundation models', rows: [
          ['Qwen3.5-9B (w/o Think)', 86.3, 84.7, 82.5, 70.0, 46.0, 34.7, 57.2, 75.4],
          ['Qwen3.5-9B (w/ Think)', 88.4, 88.7, 85.1, 72.4, 52.3, 35.0, 54.8, 80.5],
          ['InternVL3.5-8B', 70.2, 69.7, 70.1, 59.5, 41.0, 28.7, 56.3, 67.5],
          ['Cambrian-S-7B', 70.6, 64.0, 53.7, null, 39.0, 27.1, 67.5, 64.8],
          ['LLaVA-OneVision-2-8B', 72.0, 72.0, 63.1, 63.5, 43.3, 29.6, 70.9, 69.7],
          ['JoyAI-Image-Und-8B', 85.9, 78.8, 76.5, 69.6, 45.0, 35.8, 60.1, 74.0],
          ['AVTR-9B (Ours)', 89.5, 89.4, 85.1, 75.2, 54.2, 39.8, 71.0, 81.5],
        ] },
      ],
    },
    video: {
      groups: [
        { label: 'Video Understanding (Acc)', span: 4 },
        { label: 'Temporal Grounding (mIoU)', span: 4 },
      ],
      cols: ['MVBench', 'Video-MME', 'LongVideoBench', 'MLVU', 'Charades-STA', 'QVHighlights', 'ActivityNet-Cap.', 'NExT-GQA'],
      sections: [
        { label: null, rows: [
          ['Qwen3.5-9B (w/o Think)', 69.1, 72.4, 67.8, 79.1, 61.8, 67.2, 49.8, 38.4],
          ['Qwen3.5-9B (w/ Think)', 71.8, 75.4, 69.1, 80.0, 55.2, 63.5, 42.1, 34.4],
          ['InternVL3.5-8B', 72.1, 66.0, 62.1, 70.2, 27.8, 31.3, 31.3, 25.1],
          ['Video-R1', 65.5, 61.8, null, null, null, null, null, null],
          ['Time-R1', 65.9, 54.2, 55.8, null, 58.8, null, 52.1, null],
          ['VITAL', null, 64.1, null, null, 59.9, null, 49.8, 43.0],
          ['VideoChat-R1.5', 70.6, 65.2, 61.4, 70.1, 60.6, null, 35.3, null],
          ['VideoAuto-R1-8B', 72.0, 71.7, 67.4, 78.1, 63.7, 61.6, 51.9, 44.2],
          ['LLaVA-OneVision-2-8B', 66.2, 71.9, 68.0, 76.6, 53.5, 63.3, 50.5, 35.6],
          ['AVTR-9B (Ours)', 72.4, 76.9, 70.5, 81.7, 64.7, 71.5, 53.1, 44.8],
        ] },
      ],
    },
    small: {
      groups: [
        { label: 'HighRes Image', span: 4 },
        { label: 'Visual Spatial', span: 4 },
        { label: 'Video Understanding', span: 4 },
      ],
      cols: ['V*', 'HR-4K', 'HR-8K', 'BLINK', 'ERQA', 'MMSI', 'VSI', 'RWQA', 'MVBench', 'Video-MME', 'LongVideoBench', 'MLVU'],
      sections: [
        { label: null, rows: [
          ['Gemma 4 E2B', 55.0, 56.5, 47.8, 45.9, 36.3, 27.3, 21.4, 49.7, 40.6, 56.2, 46.4, 68.8],
          ['MiniCPM-V 4.6 (1.3B)', 75.4, 71.9, 65.6, 56.7, 39.0, 28.0, 37.0, 63.3, 55.8, 58.9, 55.8, 65.2],
          ['InternVL3.5-2B', 68.1, 61.4, 58.4, 52.1, 38.8, 25.7, 54.0, 59.9, 68.2, 57.5, 52.1, 65.0],
          ['Qwen3.5-2B (w/o Think)', 82.7, 77.1, 74.0, 48.3, 34.0, 22.2, 48.0, 67.7, 58.6, 60.9, 51.9, 69.3],
          ['Qwen3.5-2B (w/ Think)', 81.7, 75.4, 73.8, 60.3, 42.8, 28.8, 31.0, 70.5, 59.6, 55.4, 49.3, 70.2],
          ['AVTR-2B (Ours)', 86.4, 77.6, 75.8, 64.4, 48.0, 32.7, 60.8, 69.0, 61.3, 62.4, 59.1, 72.3],
        ] },
      ],
    },
  },

  routes: {
    direct: {
      name: 'Direct',
      trace: [['tok-muted', '</imagine>'], ['tok-muted', '</think>'], ['tok-answer', '<answer> … </answer>']],
      desc: 'Both gates close: the model answers immediately. Cheapest route, for questions that need neither re-inspection nor multi-step inference.',
    },
    visual: {
      name: 'Visual',
      trace: [['tok-visual', '<imagine>'], ['queries', ''], ['tok-visual', '</imagine>'], ['tok-muted', '</think>'], ['tok-answer', '<answer> … </answer>']],
      desc: 'The model inserts visual queries and forms a question-conditioned visual state in one parallel pass, then answers from it. Strong on geometry, viewpoint, and fine-grained perception.',
    },
    textual: {
      name: 'Textual',
      trace: [['tok-muted', '</imagine>'], ['tok-textual', '<think>'], ['lines', ''], ['tok-textual', '</think>'], ['tok-answer', '<answer> … </answer>']],
      desc: 'A textual reasoning trace before the answer. Strong on counting, temporal ordering, and discrete relational inference, but long and slow.',
    },
    both: {
      name: 'Visual + Textual',
      trace: [['tok-visual', '<imagine>'], ['queries', ''], ['tok-visual', '</imagine>'], ['tok-textual', '<think>'], ['lines', ''], ['tok-textual', '</think>'], ['tok-answer', '<answer> … </answer>']],
      desc: 'Visual imagination first, then a textual trace that can attend to the visual state. The most expensive route, for questions that need both evidence and inference.',
    },
  },

  /* Qualitative examples (text from the paper's figures). */
  qual: {
    camera: {
      question: 'Is the camera moving left or right around the static scene?',
      options: ['(A) left', '(B) right'], gt: '(A) left', source: 'BLINK · multi-view reasoning',
      inputs: [['cam_f1.jpg', 'Frame 1'], ['cam_f2.jpg', 'Frame 2']],
      visual: { image: 'cam_imagine.jpg', answer: '(A) left', correct: true },
      textual: {
        head: 'The user wants to determine the camera movement direction between two frames. 1. Analyze Image 1: Shows a white mug with',
        omitted: '3,022 words omitted', tail: 'the direction. So (B) is the correct option.', answer: '(B) right', correct: false,
      },
    },
    similar: {
      question: 'Given three similar but different images, take the first image as reference. Which of the latter two images is most similar to the first one?',
      options: ['(A) Image 2', '(B) Image 3'], gt: '(A) Image 2', source: 'BLINK · visual similarity',
      inputs: [['rings_ref.jpg', 'Reference'], ['rings_img2.jpg', 'Image 2'], ['rings_img3.jpg', 'Image 3']],
      visual: { image: 'rings_imagine.jpg', answer: '(A) Image 2', correct: true },
      textual: {
        head: 'The user compares the reference with Images 2 and 3 to find the most similar one. The first image shows onion rings on a blue wooden',
        omitted: '2,151 words omitted', tail: 'Image 1 and Image 3 look most similar. Both feature coarse salt and golden-brown coloration. Therefore, the answer should be (B).',
        answer: '(B) Image 3', correct: false,
      },
    },
    count: {
      question: 'How many people in the image are wearing caps?',
      options: ['(A) 4', '(B) 5', '(C) 3', '(D) 2'], gt: '(C) 3', source: 'BLINK · counting',
      inputs: [['count_input.jpg', 'Input image']],
      visual: { image: 'count_imagine.jpg', answer: '(D) 2', correct: false },
      textual: {
        head: 'The user is asking for the number of people wearing a cap. 1. Person in the wheelchair: This person is wearing a blue cap.',
        omitted: '763 words omitted', tail: 'Person near yellow plane: Yes. The answer is C.', answer: '(C) 3', correct: true,
      },
    },
  },
  videoExamples: {
    video: {
      question: 'When making rice cakes, what is the last ingredient added before adding water?',
      source: 'Long-video QA · 14:19 video', framePrefix: 'rice', image: 'rice_imagine.jpg',
      times: ['00:00', '01:08', '02:36', '03:35', '04:56', '06:16', '07:37', '08:57', '10:18', '11:38', '12:59', '14:19'],
      salient: 2, answer: 'Onion', answerClass: 'ok',
    },
    ground: {
      question: 'When does a woman eat at a restaurant near the water? Give the start and end time in seconds.',
      source: 'Temporal grounding · 02:30 video', framePrefix: 'ground', image: 'ground_imagine.jpg',
      // Only the five timestamps explicitly labelled in the source figure are shown.
      times: ['00:00', null, null, '00:41', null, null, '01:22', null, null, '02:02', null, '02:30'],
      salient: 9, answer: '[110, 140] s', answerClass: '',
    },
  },
};
