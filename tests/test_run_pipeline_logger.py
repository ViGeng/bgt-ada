from run_pipeline import TeeLogger


def test_tee_logger_skips_carriage_return_progress_updates(tmp_path):
    log_path = tmp_path / "run.log"
    logger = TeeLogger(log_path)

    try:
        logger.write("pipeline start\n")
        logger.write("  Epoch 1/2:   0%|          | 0/12 [00:00<?, ?it/s]\r")
        logger.write("  Epoch 1/2: 100%|##########| 12/12 [00:06<00:00,  2.00it/s]\r")
        logger.write("pipeline done\n")
    finally:
        logger.close()

    assert log_path.read_text() == "pipeline start\npipeline done\n"