import os
import gradio as gr
import rerun as rr
import cv2
import torch
import gc
from app import (
    get_meta_from_img_seq,
    get_meta_from_video,
    init_SegTracker,
    clean,
    init_SegTracker_Stroke,
    segment_everything,
    sam_click,
    sam_stroke,
    gd_detect,
    add_new_object,
    tracking_objects,
    undo_click_stack_and_refine_seg,
)


def tracking_objects(Seg_Tracker, input_video):
    return video_type_input_tracking(Seg_Tracker, input_video)


def video_type_input_tracking(SegTracker, input_video: str):
    print("Start tracking !")
    # source video to segment
    cap = cv2.VideoCapture(input_video)

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0
    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rr.set_time_sequence("frame_idx", frame_idx)
            rr.log_image("image/rgb", frame)
            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
                SegTracker.add_reference(frame, pred_mask)
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask, seg_mask)
                pred_mask = track_mask + new_obj_mask
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            rr.log_segmentation_image("image/pred_mask", pred_mask)

            print(
                "processed frame {}, obj_num {}".format(
                    frame_idx, SegTracker.get_obj_num()
                ),
                end="\r",
            )
            frame_idx += 1
        cap.release()
        print("\nfinished")


def main():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    block = gr.Blocks()

    with block:
        gr.Markdown(
            """
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Segment and Track Anything(SAM-Track) visualized with Rerun</span>
            </div>
            """
        )

        click_stack = gr.State([[], []])  # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row():
            with gr.Column(scale=0.35):
                tab_video_input = gr.Tab(label="Video type input")
                with tab_video_input:
                    input_video = gr.Video(label="Input video").style(height=550)

                tab_img_seq_input = gr.Tab(label="Image-Seq type input")
                with tab_img_seq_input:
                    with gr.Row():
                        input_img_seq = gr.File(label="Input Image-Seq").style(
                            height=550
                        )
                        with gr.Column(scale=0.25):
                            extract_button = gr.Button(value="extract")
                            fps = gr.Slider(
                                label="fps", minimum=5, maximum=50, value=8, step=1
                            )

            with gr.Column(scale=0.65):
                # Preprocessing
                gr.HTML(
                    # value='<iframe src="http://127.0.0.1:9090/?url=ws://localhost:9877" width="950" height="712"></iframe>'
                    value='<iframe src="https://app.rerun.io/" width="950" height="712"></iframe>'
                )
        with gr.Row():
            input_first_frame = gr.Image(
                label="Segment result of first frame", interactive=True
            ).style(height=550)

            with gr.Column():
                tab_everything = gr.Tab(label="Everything")
                with tab_everything:
                    with gr.Row():
                        seg_every_first_frame = gr.Button(
                            value="Segment everything for first frame", interactive=True
                        )
                        point_mode = gr.Radio(
                            choices=["Positive"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True,
                        )

                        every_undo_but = gr.Button(value="Undo", interactive=True)
                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        point_mode = gr.Radio(
                            choices=["Positive", "Negative"],
                            value="Positive",
                            label="Point Prompt",
                            interactive=True,
                        )

                        # args for modify and tracking
                        click_undo_but = gr.Button(value="Undo", interactive=True)

                tab_stroke = gr.Tab(label="Stroke")
                with tab_stroke:
                    drawing_board = gr.Image(
                        label="Drawing Board",
                        tool="sketch",
                        brush_radius=10,
                        interactive=True,
                    )
                    with gr.Row():
                        seg_acc_stroke = gr.Button(value="Segment", interactive=True)

                tab_text = gr.Tab(label="Text")
                with tab_text:
                    grounding_caption = gr.Textbox(label="Detection Prompt")
                    detect_button = gr.Button(value="Detect")
                    with gr.Accordion("Advanced options", open=False):
                        with gr.Row():
                            with gr.Column(scale=0.5):
                                box_threshold = gr.Slider(
                                    label="Box Threshold",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.25,
                                    step=0.001,
                                )
                            with gr.Column(scale=0.5):
                                text_threshold = gr.Slider(
                                    label="Text Threshold",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.25,
                                    step=0.001,
                                )
                with gr.Column():
                    new_object_button = gr.Button(
                        value="Add new object", interactive=True
                    )
                    reset_button = gr.Button(
                        value="Reset",
                        interactive=True,
                    )
                    track_for_video = gr.Button(
                        value="Start Tracking",
                        interactive=True,
                    ).style(size="lg")
            with gr.Row():
                with gr.Column(scale=0.5):
                    with gr.Tab(label="SegTracker Args"):
                        # args for tracking in video do segment-everthing
                        points_per_side = gr.Slider(
                            label="points_per_side",
                            minimum=1,
                            step=1,
                            maximum=100,
                            value=16,
                            interactive=True,
                        )

                        sam_gap = gr.Slider(
                            label="sam_gap",
                            minimum=1,
                            step=1,
                            maximum=9999,
                            value=100,
                            interactive=True,
                        )

                        max_obj_num = gr.Slider(
                            label="max_obj_num",
                            minimum=50,
                            step=1,
                            maximum=300,
                            value=255,
                            interactive=True,
                        )
                        with gr.Accordion("aot advanced options", open=False):
                            aot_model = gr.Dropdown(
                                label="aot_model",
                                choices=["deaotb", "deaotl", "r50_deaotl"],
                                value="r50_deaotl",
                                interactive=True,
                            )
                            long_term_mem = gr.Slider(
                                label="long term memory gap",
                                minimum=1,
                                maximum=9999,
                                value=9999,
                                step=1,
                            )
                            max_len_long_term = gr.Slider(
                                label="max len of long term memory",
                                minimum=1,
                                maximum=9999,
                                value=9999,
                                step=1,
                            )
        ##########################################################
        ######################  back-end #########################
        ##########################################################

        # listen to the input_video to get the first frame of video
        input_video.change(
            fn=get_meta_from_video,
            inputs=[input_video],
            outputs=[input_first_frame, origin_frame, drawing_board, grounding_caption],
        )

        # listen to the input_img_seq to get the first frame of video
        input_img_seq.change(
            fn=get_meta_from_img_seq,
            inputs=[input_img_seq],
            outputs=[input_first_frame, origin_frame, drawing_board, grounding_caption],
        )

        # -------------- Input compont -------------
        tab_video_input.select(
            fn=clean,
            inputs=[],
            outputs=[
                input_video,
                input_img_seq,
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                drawing_board,
                click_stack,
            ],
        )

        tab_img_seq_input.select(
            fn=clean,
            inputs=[],
            outputs=[
                input_video,
                input_img_seq,
                Seg_Tracker,
                input_first_frame,
                origin_frame,
                drawing_board,
                click_stack,
            ],
        )

        extract_button.click(
            fn=get_meta_from_img_seq,
            inputs=[input_img_seq],
            outputs=[input_first_frame, origin_frame, drawing_board],
        )

        # ------------------- Interactive component -----------------

        # listen to the tab to init SegTracker
        tab_everything.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption],
            queue=False,
        )

        tab_click.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption],
            queue=False,
        )

        tab_stroke.select(
            fn=init_SegTracker_Stroke,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack, drawing_board],
            queue=False,
        )

        tab_text.select(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption],
            queue=False,
        )

        # Use SAM to segment everything for the first frame of video
        seg_every_first_frame.click(
            fn=segment_everything,
            inputs=[
                Seg_Tracker,
                aot_model,
                long_term_mem,
                max_len_long_term,
                origin_frame,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[
                Seg_Tracker,
                input_first_frame,
            ],
        )

        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=sam_click,
            inputs=[
                Seg_Tracker,
                origin_frame,
                point_mode,
                click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack],
        )

        # Interactively segment acc stroke
        seg_acc_stroke.click(
            fn=sam_stroke,
            inputs=[
                Seg_Tracker,
                origin_frame,
                drawing_board,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[Seg_Tracker, input_first_frame, drawing_board],
        )

        # Use grounding-dino to detect object
        detect_button.click(
            fn=gd_detect,
            inputs=[
                Seg_Tracker,
                origin_frame,
                grounding_caption,
                box_threshold,
                text_threshold,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[Seg_Tracker, input_first_frame],
        )

        # Add new object
        new_object_button.click(
            fn=add_new_object, inputs=[Seg_Tracker], outputs=[Seg_Tracker, click_stack]
        )

        # Track object in video
        track_for_video.click(
            fn=tracking_objects,
            inputs=[
                Seg_Tracker,
                input_video,
                # input_img_seq,
                # fps,
            ],
            # outputs=[output_video, output_mask],
        )

        # ----------------- Reset and Undo ---------------------------

        reset_button.click(
            fn=init_SegTracker,
            inputs=[
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
                origin_frame,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack, grounding_caption],
            queue=False,
            show_progress=False,
        )

        # Undo click
        click_undo_but.click(
            fn=undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker,
                origin_frame,
                click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack],
        )

        every_undo_but.click(
            fn=undo_click_stack_and_refine_seg,
            inputs=[
                Seg_Tracker,
                origin_frame,
                click_stack,
                aot_model,
                long_term_mem,
                max_len_long_term,
                sam_gap,
                max_obj_num,
                points_per_side,
            ],
            outputs=[Seg_Tracker, input_first_frame, click_stack],
        )

        with gr.Tab(label="Video example"):
            gr.Examples(
                examples=[
                    # os.path.join(os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "blackswan.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "Resized_cxk.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "bear.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "camel.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "skate-park.mp4"),
                    # os.path.join(os.path.dirname(__file__), "assets", "swing.mp4"),
                ],
                inputs=[input_video],
            )

        with gr.Tab(label="Image-seq expamle"):
            gr.Examples(
                examples=[
                    os.path.join(
                        os.path.dirname(__file__), "assets", "840_iSXIa0hE8Ek.zip"
                    ),
                ],
                inputs=[input_img_seq],
            )
    block.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    rr.init(application_id="application_id", default_enabled=True, strict=True)
    rr.serve(open_browser=False)
    main()
    rr.disconnect()
