import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os
import time

load_dotenv()

SOURCE_BUCKET = 'leasedocs-1'
DESTINATION_BUCKET = 'textract-out-files'
AWS_REGION = 'us-west-2'
FILE_EXTENSION = '.pdf'

TARGET_PREFIX = 'FL Babcock/'    # e.g. 'folder/subfolder/' or '' for whole bucket
OVERWRITE = False                    # set True to re-generate .txt even if it exists

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 60 * 30

def s3_put_text(s3_client, bucket, key, text):
    s3_client.put_object(Bucket=bucket, Key=key, Body=text.encode('utf-8'), ContentType='text/plain')

def extract_lines_from_blocks(blocks):
    return [b['Text'] for b in blocks if b.get('BlockType') == 'LINE' and 'Text' in b]

def txt_exists(s3_client, bucket, txt_key):
    try:
        s3_client.head_object(Bucket=bucket, Key=txt_key)
        return True
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise

def process_job_to_txt(textract_client, s3_client, job_id, source_key, dest_bucket):
    print(f"Waiting for job {job_id} to complete...")
    start = time.time()
    pages = []
    next_token = None

    while True:
        if time.time() - start > POLL_TIMEOUT_SECONDS:
            raise TimeoutError(f"Timed out waiting for Textract job {job_id}.")

        try:
            kwargs = {'JobId': job_id, 'MaxResults': 1000}
            if next_token:
                kwargs['NextToken'] = next_token
            resp = textract_client.get_document_text_detection(**kwargs)
        except ClientError as e:
            code = e.response.get('Error', {}).get('Code')
            if code in ('InvalidJobIdException', 'ThrottlingException'):
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            raise

        status = resp.get('JobStatus')
        if status in ('IN_PROGRESS', 'SUCCEEDED'):
            blocks = resp.get('Blocks', []) or []
            pages.append(blocks)
            next_token = resp.get('NextToken')

            if next_token:
                continue
            if status == 'IN_PROGRESS':
                pages = []  # discard partials; SUCCEEDED will re-send full results
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            break
        elif status in ('FAILED', 'PARTIAL_SUCCESS'):
            print(f"Job {job_id} ended with status {status}.")
            return False
        else:
            time.sleep(POLL_INTERVAL_SECONDS)

    print(f"✅ Job {job_id} SUCCEEDED.")
    all_blocks = [b for blk_list in pages for b in blk_list]
    lines = extract_lines_from_blocks(all_blocks)

    base, _ = os.path.splitext(source_key)
    txt_key = f"{base}.txt"
    txt_content = "\n".join(lines)
    s3_put_text(s3_client, dest_bucket, txt_key, txt_content)

    print(f"Wrote text to s3://{dest_bucket}/{txt_key} ({len(lines)} lines, {len(txt_content)} chars)")
    return True

def process_s3_documents(source_bucket, destination_bucket, region, file_extension, target_prefix):
    s3_client = boto3.client('s3', region_name=region)
    textract_client = boto3.client('textract', region_name=region)
    paginator = s3_client.get_paginator('list_objects_v2')

    if target_prefix and not target_prefix.endswith('/'):
        target_prefix += '/'

    prefix_msg = target_prefix or '(entire bucket)'
    print(f"Searching for '{file_extension}' files in s3://{source_bucket}/{prefix_msg} ...")

    # Only list objects under the target prefix (single folder)
    pages = paginator.paginate(Bucket=source_bucket, Prefix=target_prefix) if target_prefix else paginator.paginate(Bucket=source_bucket)

    started = []
    for page in pages:
        for obj in page.get('Contents', []):
            object_key = obj['Key']
            if not object_key.lower().endswith(file_extension) or object_key.endswith('/'):
                continue

            # Skip if .txt already exists and OVERWRITE is False
            base, _ = os.path.splitext(object_key)
            txt_key = f"{base}.txt"
            if not OVERWRITE and txt_exists(s3_client, destination_bucket, txt_key):
                print(f"⏭Skipping (already converted): s3://{destination_bucket}/{txt_key}")
                continue

            output_prefix = os.path.dirname(object_key)
            if output_prefix and not output_prefix.endswith('/'):
                output_prefix += '/'

            print(f"\n--- Starting Textract for: {object_key} ---")
            try:
                response = textract_client.start_document_text_detection(
                    DocumentLocation={'S3Object': {'Bucket': source_bucket, 'Name': object_key}},
                    OutputConfig={'S3Bucket': destination_bucket, 'S3Prefix': output_prefix}
                )
                job_id = response['JobId']
                print(f"📤 Job started: {job_id}")
                print(f"   JSON output: s3://{destination_bucket}/{output_prefix}{job_id}/")
                started.append((job_id, object_key))
            except Exception as e:
                print(f"ERROR starting job for {object_key}: {e}")

    print(f"\nStarted {len(started)} Textract jobs under prefix '{target_prefix or '/'}'.")

    successes = 0
    for job_id, object_key in started:
        try:
            ok = process_job_to_txt(textract_client, s3_client, job_id, object_key, destination_bucket)
            successes += 1 if ok else 0
        except Exception as e:
            print(f"ERROR processing job {job_id} ({object_key}): {e}")

    print(f"\nSummary: {successes}/{len(started)} documents converted to .txt.")

if __name__ == '__main__':
    if SOURCE_BUCKET == 'your-source-pdf-bucket' or DESTINATION_BUCKET == 'your-textract-output-bucket':
        print("ERROR: Please update the SOURCE_BUCKET and DESTINATION_BUCKET.")
    else:
        process_s3_documents(
            source_bucket=SOURCE_BUCKET,
            destination_bucket=DESTINATION_BUCKET,
            region=AWS_REGION,
            file_extension=FILE_EXTENSION,
            target_prefix=TARGET_PREFIX
        )
